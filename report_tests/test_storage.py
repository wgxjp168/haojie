import json
from pathlib import Path

import pytest

from report.core.exceptions import ReportNotFoundError
from report.storage import LocalFilesystemBackend, MemoryStorageBackend, S3StorageBackend


# -------------------- memory --------------------


def test_memory_backend_always_available():
    b = MemoryStorageBackend()
    assert b.available is True


def test_memory_put_get_roundtrip():
    b = MemoryStorageBackend()
    loc = b.put(
        "r1",
        b"hello world",
        content_type="text/plain; charset=utf-8",
        checksum="abc123",
    )
    assert loc.startswith("memory://")
    obj = b.get("r1")
    assert obj.content == b"hello world"
    assert obj.content_type.startswith("text/plain")
    assert obj.checksum == "abc123"


def test_memory_get_missing_raises():
    b = MemoryStorageBackend()
    with pytest.raises(ReportNotFoundError):
        b.get("nope")


def test_memory_exists_and_delete():
    b = MemoryStorageBackend()
    b.put("r1", b"data", content_type="text/plain", checksum="x")
    assert b.exists("r1")
    assert b.delete("r1")
    assert not b.exists("r1")
    assert b.delete("r1") is False


def test_memory_capacity_evicts_oldest():
    b = MemoryStorageBackend(capacity=2)
    b.put("a", b"1", content_type="text/plain", checksum="")
    b.put("b", b"2", content_type="text/plain", checksum="")
    b.put("c", b"3", content_type="text/plain", checksum="")
    assert not b.exists("a")
    assert b.exists("b")
    assert b.exists("c")


# -------------------- local --------------------


def test_local_backend_creates_base_dir(temp_dir):
    b = LocalFilesystemBackend(base_dir=str(Path(temp_dir) / "reports"))
    assert b.available is True
    assert (Path(temp_dir) / "reports").exists()


def test_local_put_creates_sidecar(temp_dir):
    b = LocalFilesystemBackend(base_dir=temp_dir)
    loc = b.put(
        "r1",
        b"# hi",
        content_type="text/markdown; charset=utf-8",
        checksum="cafe",
        metadata={"language": "zh-CN"},
    )
    assert loc.startswith("file://")
    sidecar = Path(temp_dir) / "r1.meta.json"
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["checksum"] == "cafe"
    assert meta["content_type"].startswith("text/markdown")


def test_local_get_roundtrip(temp_dir):
    b = LocalFilesystemBackend(base_dir=temp_dir)
    b.put(
        "r1",
        b"# hi",
        content_type="text/markdown; charset=utf-8",
        checksum="cafe",
    )
    obj = b.get("r1")
    assert obj.content == b"# hi"
    assert obj.content_type.startswith("text/markdown")
    assert obj.checksum == "cafe"


def test_local_get_missing_raises(temp_dir):
    b = LocalFilesystemBackend(base_dir=temp_dir)
    with pytest.raises(ReportNotFoundError):
        b.get("ghost")


def test_local_safe_id_escapes(temp_dir):
    b = LocalFilesystemBackend(base_dir=temp_dir)
    loc = b.put(
        "../../etc/passwd",
        b"x",
        content_type="text/plain; charset=utf-8",
        checksum="",
    )
    # Escaped in the filename.
    assert "passwd" not in Path(loc.replace("file://", "")).name or "_" in Path(
        loc.replace("file://", "")
    ).name


def test_local_delete_removes_both_files(temp_dir):
    b = LocalFilesystemBackend(base_dir=temp_dir)
    b.put("r1", b"x", content_type="text/plain; charset=utf-8", checksum="")
    assert b.delete("r1") is True
    assert not (Path(temp_dir) / "r1.txt").exists()
    assert not (Path(temp_dir) / "r1.meta.json").exists()


# -------------------- s3 --------------------


def test_s3_backend_unavailable_without_boto3():
    # boto3 is not installed in the test env.
    b = S3StorageBackend(bucket="x", region="us-east-1")
    assert b.available is False


def test_s3_exists_on_unavailable_returns_false():
    b = S3StorageBackend(bucket="x")
    assert b.exists("r1") is False


def test_s3_delete_on_unavailable_returns_false():
    b = S3StorageBackend(bucket="x")
    assert b.delete("r1") is False
