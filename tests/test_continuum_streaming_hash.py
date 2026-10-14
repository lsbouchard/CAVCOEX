"""Bounded archive hashing preserves exact recorded digests."""

import hashlib
import pytest

from scripts.continuum_replica_exchange import file_sha256


@pytest.mark.parametrize("chunk_size", [1, 1024, 1024*1024])
@pytest.mark.parametrize("content", [b"", b"cavity"*40001])
def test_chunked_digest_matches_exact_bytes(tmp_path,chunk_size,content):
    path = tmp_path/"small_archive"
    path.write_bytes(content)
    assert file_sha256(path,chunk_size) == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("chunk_size", [0,-1,1.5,None])
def test_invalid_chunk_size_rejected(tmp_path,chunk_size):
    with pytest.raises(ValueError,match="positive integer"):
        file_sha256(tmp_path/"not_read",chunk_size)
