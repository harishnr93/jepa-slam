"""Selective, verified download of KITTI Odometry from the official archives.

The official colour archive is a single 69 GB zip containing all 22 sequences and both
colour cameras. Loop-closure work only needs a few sequences from one camera, so this
module downloads *individual zip members* using HTTP range requests:

1. Read the zip central directory (at the end of the file) with range requests.
2. Select members, e.g. ``dataset/sequences/00/image_2/*.png``.
3. Group selected members into contiguous byte spans and stream each span once,
   parsing local file headers on the fly (one request per span, not per file).
4. Decompress if needed, verify the CRC-32 stored in the central directory and write
   each file atomically (temporary name + rename). Existing files of the correct size
   are skipped, so interrupted downloads resume.

Small archives (calibration, poses) are handled the same way. Nothing here is specific
to HTTP: ``RangeSource`` abstracts "read bytes [start, end)", which the tests implement
with a local file.
"""

from __future__ import annotations

import fnmatch
import io
import os
import struct
import time
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol

from latent_loop.utils.logging import get_logger

logger = get_logger(__name__)

KITTI_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
ARCHIVES = {
    "color": "data_odometry_color.zip",
    "gray": "data_odometry_gray.zip",
    "calib": "data_odometry_calib.zip",
    "poses": "data_odometry_poses.zip",
}
# Sequences with revisits (loop closures) in the KITTI Odometry training split.
LOOP_SEQUENCES = ("00", "02", "05", "06", "08", "09")
_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")  # 30 bytes
_LOCAL_SIG = b"PK\x03\x04"


class RangeSource(Protocol):
    size: int

    def open_range(self, start: int, end: int) -> BinaryIO:
        """Return a stream of bytes ``[start, end)``."""


class HttpRangeSource:
    def __init__(self, url: str, timeout: float = 60.0, retries: int = 5) -> None:
        self.url, self.timeout, self.retries = url, timeout, retries
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.headers.get("Accept-Ranges", "") != "bytes":
                raise RuntimeError(f"{url} does not support HTTP range requests")
            self.size = int(response.headers["Content-Length"])

    def open_range(self, start: int, end: int) -> BinaryIO:
        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{end - 1}"}
                )
                response = urllib.request.urlopen(request, timeout=self.timeout)
                if response.status != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status}")
                return response
            except OSError as exc:
                if attempt == self.retries:
                    raise
                logger.warning(
                    "Range request failed; retrying", extra={"attempt": attempt, "error": str(exc)}
                )
                time.sleep(2 * attempt)
        raise AssertionError("unreachable")


class FileRangeSource:
    """Local-file implementation of :class:`RangeSource` (tests, local archives)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.size = self.path.stat().st_size

    def open_range(self, start: int, end: int) -> BinaryIO:
        with open(self.path, "rb") as handle:
            handle.seek(start)
            return io.BytesIO(handle.read(end - start))


class _SeekableRange(io.RawIOBase):
    """Read-only seekable view over a RangeSource with a read-ahead buffer.

    Only used to let :mod:`zipfile` parse the central directory.
    """

    def __init__(self, source: RangeSource, block: int = 1 << 20) -> None:
        self.source, self.block, self.pos = source, block, 0
        self._buf_start, self._buf = 0, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.source.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.source.size - self.pos)
        if n <= 0:
            return 0
        end = self.pos + n
        if not (self._buf_start <= self.pos and end <= self._buf_start + len(self._buf)):
            start = self.pos
            stop = min(self.source.size, max(end, start + self.block))
            self._buf_start, self._buf = start, self.source.open_range(start, stop).read()
        off = self.pos - self._buf_start
        buffer[:n] = self._buf[off : off + n]
        self.pos = end
        return n


def list_members(source: RangeSource) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(io.BufferedReader(_SeekableRange(source))) as archive:
        return archive.infolist()


def select_members(
    members: Iterable[zipfile.ZipInfo], patterns: Iterable[str]
) -> list[zipfile.ZipInfo]:
    patterns = list(patterns)
    return [
        m
        for m in members
        if not m.is_dir() and any(fnmatch.fnmatch(m.filename, p) for p in patterns)
    ]


def image_patterns(sequences: Iterable[str], cameras: Iterable[int]) -> list[str]:
    return [f"dataset/sequences/{s}/image_{c}/*.png" for s in sequences for c in cameras]


@dataclass
class DownloadReport:
    selected: int = 0
    skipped_existing: int = 0
    downloaded: int = 0
    bytes_downloaded: int = 0
    crc_verified: int = 0
    seconds: float = 0.0
    files: list[str] = field(default_factory=list)


def _target(out_root: Path, member: zipfile.ZipInfo, strip_prefix: str) -> Path:
    """Output path for a member. Validated as a string, without file-system calls:
    on WSL ``/mnt/<drive>`` every path lookup is a slow 9P round trip."""
    name = member.filename
    if strip_prefix and name.startswith(strip_prefix):
        name = name[len(strip_prefix) :]
    parts = PurePosixPath(name).parts
    if not parts or name.startswith(("/", "\\")) or ".." in parts or ":" in parts[0]:
        raise ValueError(f"unsafe path in archive: {member.filename}")
    return out_root.joinpath(*parts)


def _existing_sizes(paths: Iterable[Path]) -> dict[Path, int]:
    """Sizes of existing files, listing each parent directory once (cheap on 9P mounts)."""
    sizes: dict[Path, int] = {}
    for directory in {p.parent for p in paths}:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file():
                        sizes[directory / entry.name] = entry.stat().st_size
        except FileNotFoundError:
            continue
    return sizes


def _spans(members: list[zipfile.ZipInfo], max_gap: int) -> Iterator[list[zipfile.ZipInfo]]:
    """Group members (sorted by offset) into runs whose byte gaps are <= ``max_gap``."""
    run: list[zipfile.ZipInfo] = []
    end = -1
    for m in sorted(members, key=lambda m: m.header_offset):
        start = m.header_offset
        if run and start - end > max_gap:
            yield run
            run = []
        run.append(m)
        # Upper bound of the member's end: local header + name + generous extra field.
        end = start + 30 + len(m.filename.encode()) + 1024 + m.compress_size
    if run:
        yield run


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    chunks, remaining = [], n
    while remaining:
        chunk = stream.read(min(remaining, 1 << 20))
        if not chunk:
            raise EOFError(f"stream ended {remaining} bytes early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_member(stream: BinaryIO, member: zipfile.ZipInfo, target: Path) -> int:
    header = _read_exact(stream, _LOCAL_HEADER.size)
    sig, *_rest, name_len, extra_len = _LOCAL_HEADER.unpack(header)
    if sig != _LOCAL_SIG:
        raise ValueError(f"bad local header signature for {member.filename}")
    name = _read_exact(stream, name_len).decode("utf-8", "replace")
    if name != member.filename:
        raise ValueError(f"local header name {name!r} != central directory {member.filename!r}")
    _read_exact(stream, extra_len)
    data = _read_exact(stream, member.compress_size)
    if member.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif member.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"unsupported compression {member.compress_type} in {member.filename}")
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        raise ValueError(f"CRC/size mismatch for {member.filename}")
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return 30 + name_len + extra_len + member.compress_size


def download_members(
    source: RangeSource,
    members: list[zipfile.ZipInfo],
    out_root: str | Path,
    strip_prefix: str = "",
    max_gap: int = 4 << 20,
    progress: Callable[[DownloadReport], None] | None = None,
    progress_every: int = 500,
) -> DownloadReport:
    """Download, verify and write ``members`` below ``out_root`` (resumable)."""
    out_root = Path(out_root)
    report = DownloadReport(selected=len(members))
    start_time = time.perf_counter()
    targets = {m.filename: _target(out_root, m, strip_prefix) for m in members}
    sizes = _existing_sizes(targets.values())
    todo = []
    for m in members:
        if sizes.get(targets[m.filename]) == m.file_size:
            report.skipped_existing += 1
        else:
            todo.append(m)
    for directory in {targets[m.filename].parent for m in todo}:  # once per directory
        directory.mkdir(parents=True, exist_ok=True)
    for run in _spans(todo, max_gap):
        first = run[0].header_offset
        last = run[-1]
        stop = min(
            source.size,
            last.header_offset + 30 + len(last.filename.encode()) + 1024 + last.compress_size,
        )
        stream = source.open_range(first, stop)
        pos = first
        try:
            for m in run:
                if m.header_offset > pos:  # skip unselected bytes between members
                    _read_exact(stream, m.header_offset - pos)
                    pos = m.header_offset
                pos += _write_member(stream, m, targets[m.filename])
                report.downloaded += 1
                report.crc_verified += 1
                report.bytes_downloaded += m.compress_size
                if progress and report.downloaded % progress_every == 0:
                    report.seconds = time.perf_counter() - start_time
                    progress(report)
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
    report.seconds = time.perf_counter() - start_time
    return report
