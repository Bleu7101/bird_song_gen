from __future__ import annotations

from pathlib import Path

from bird_song.evaluation.provenance import git_revision, sha256_file


def test_sha256_file_returns_content_digest(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"hello world\n")

    assert sha256_file(path) == (
        "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
    )


def test_git_revision_returns_unknown_outside_a_repository(tmp_path: Path) -> None:
    assert git_revision(tmp_path) == "unknown"
