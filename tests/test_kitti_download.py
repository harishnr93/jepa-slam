"""Selective zip-member download against a local archive (no network)."""

from __future__ import annotations

import zipfile

import pytest

from latent_loop.datasets.kitti_download import (
    FileRangeSource,
    download_members,
    image_patterns,
    list_members,
    select_members,
)


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "odometry.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for seq in ("00", "01", "05"):
            for cam in (2, 3):
                for i in range(4):
                    data = bytes([int(seq), cam, i]) * (100 + i)
                    method = zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED
                    zf.writestr(
                        f"dataset/sequences/{seq}/image_{cam}/{i:06d}.png",
                        data,
                        compress_type=method,
                    )
            zf.writestr(f"dataset/sequences/{seq}/calib.txt", f"P0: {seq}\n")
        zf.writestr("dataset/poses/00.txt", "1 0 0 0 0 1 0 0 0 0 1 0\n")
    return path


def test_list_and_select(archive):
    members = list_members(FileRangeSource(archive))
    assert len(members) == 3 * 2 * 4 + 3 + 1
    selected = select_members(members, image_patterns(["00", "05"], [2]))
    assert sorted(m.filename for m in selected)[0] == "dataset/sequences/00/image_2/000000.png"
    assert len(selected) == 8


def test_download_verifies_and_resumes(archive, tmp_path):
    source = FileRangeSource(archive)
    members = select_members(list_members(source), image_patterns(["00", "05"], [2]))
    out = tmp_path / "out"
    report = download_members(source, members, out, max_gap=10)  # forces several spans
    assert report.downloaded == report.crc_verified == 8 and report.skipped_existing == 0
    with zipfile.ZipFile(archive) as zf:
        for m in members:
            assert (out / m.filename).read_bytes() == zf.read(m.filename)
    assert not (out / "dataset/sequences/01").exists()  # unselected sequence not written
    (out / members[0].filename).unlink()
    again = download_members(source, members, out)
    assert again.downloaded == 1 and again.skipped_existing == 7


def test_corrupted_archive_is_rejected(archive, tmp_path):
    source = FileRangeSource(archive)
    members = select_members(list_members(source), ["dataset/poses/*.txt"])
    blob = bytearray(archive.read_bytes())
    m = members[0]
    data_start = m.header_offset + 30 + len(m.filename) + len(m.extra)
    blob[data_start] ^= 0xFF  # flip one payload byte
    bad = tmp_path / "bad.zip"
    bad.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="CRC"):
        download_members(FileRangeSource(bad), members, tmp_path / "out")
    assert not list((tmp_path / "out").rglob("*.txt"))  # nothing half-written


def test_unsafe_member_names_are_rejected(tmp_path):
    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("dataset/../../escape.txt", "x")
    source = FileRangeSource(path)
    with pytest.raises(ValueError, match="unsafe"):
        download_members(source, list_members(source), tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()
