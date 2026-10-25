"""Stable checksum reuse must not change bytes, suffix policy or source checks."""
from collections import OrderedDict
import concurrent.futures
import gzip
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from madgraph.various import macnlops as mc


class TestMacNLOPSHashCache(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='macnlops-hash-cache-')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        with mc._checksum_cache_lock:
            mc._checksum_cache.clear()
        self.addCleanup(self.clear_cache)

    @staticmethod
    def clear_cache():
        with mc._checksum_cache_lock:
            mc._checksum_cache.clear()

    def test_plain_hashes_match_bytes_and_reuse_a_stable_inode(self):
        path = self.folder / 'events.lhe'
        content = b'plain LHE\r\nincluding non-UTF8 bytes \xff and trailing data\n'
        path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        with patch.object(mc, '_read_checksum', wraps=mc._read_checksum) as reads:
            self.assertEqual(mc.sha256(path), expected)
            self.assertEqual(mc.sha256(str(path)), expected)
            self.assertEqual(mc.lhe_sha256(path), expected)
        self.assertEqual(reads.call_count, 1)

    def test_raw_and_decoded_gzip_digests_use_separate_cache_entries(self):
        content = b'<LesHouchesEvents>\r\n<event>data</event>\r\n'
        path = self.folder / 'events.lhe.gz'
        compressed = gzip.compress(content, mtime=1)
        path.write_bytes(compressed)
        with patch.object(mc, '_read_checksum', wraps=mc._read_checksum) as reads:
            for unused in range(3):
                self.assertEqual(mc.sha256(path), hashlib.sha256(compressed).hexdigest())
                self.assertEqual(mc.lhe_sha256(path), hashlib.sha256(content).hexdigest())
        self.assertEqual(reads.call_count, 2)

    def test_suffix_and_aliases_preserve_existing_open_lhe_semantics(self):
        path = self.folder / 'events.lhe.gz'
        content = b'compressed event stream'
        compressed = gzip.compress(content, mtime=1)
        path.write_bytes(compressed)
        alias = self.folder / 'raw_alias.lhe'
        upper = self.folder / 'upper.GZ'
        os.link(path, alias)
        os.link(path, upper)
        raw = hashlib.sha256(compressed).hexdigest()
        decoded = hashlib.sha256(content).hexdigest()
        self.assertEqual(mc.lhe_sha256(path), decoded)
        self.assertEqual(mc.lhe_sha256(alias), raw)
        self.assertEqual(mc.lhe_sha256(upper), raw)
        # open_lhe has always checked str(path), including its bytes-path
        # spelling. Do not silently broaden its gzip-suffix policy here.
        self.assertEqual(mc.lhe_sha256(os.fsencode(path)), raw)

    def test_invalid_gzip_cannot_use_a_cached_raw_digest(self):
        path = self.folder / 'invalid.lhe.gz'
        content = b'not a gzip stream'
        path.write_bytes(content)
        self.assertEqual(mc.sha256(path), hashlib.sha256(content).hexdigest())
        with self.assertRaises(gzip.BadGzipFile):
            mc.lhe_sha256(path)

    def test_hardlinks_and_symlinks_share_only_stable_content(self):
        path = self.folder / 'original'
        path.write_bytes(b'original payload')
        hard = self.folder / 'hard'
        soft = self.folder / 'soft'
        os.link(path, hard)
        soft.symlink_to(path)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(mc, '_read_checksum', wraps=mc._read_checksum) as reads:
            for candidate in [path, hard, soft]:
                self.assertEqual(mc.sha256(candidate), expected)
        self.assertEqual(reads.call_count, 1)
        replacement = self.folder / 'other'
        replacement.write_bytes(b'different payload')
        soft.unlink()
        soft.symlink_to(replacement)
        self.assertEqual(mc.sha256(soft), hashlib.sha256(replacement.read_bytes()).hexdigest())

    def test_rewrite_with_restored_size_and_mtime_invalidates_by_ctime(self):
        path = self.folder / 'source'
        path.write_bytes(b'first')
        before = path.stat()
        with patch.object(mc, '_read_checksum', wraps=mc._read_checksum) as reads:
            self.assertEqual(mc.sha256(path), hashlib.sha256(b'first').hexdigest())
            path.write_bytes(b'later')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertEqual(path.stat().st_size, before.st_size)
            self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(mc.sha256(path), hashlib.sha256(b'later').hexdigest())
        self.assertEqual(reads.call_count, 2)

    def test_atomic_replacement_invalidates_even_with_restored_mtime(self):
        path = self.folder / 'source'
        path.write_bytes(b'first')
        before = path.stat()
        mc.sha256(path)
        replacement = self.folder / 'replacement'
        replacement.write_bytes(b'later')
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
        self.assertNotEqual(path.stat().st_ino, before.st_ino)
        self.assertEqual(mc.sha256(path), hashlib.sha256(b'later').hexdigest())

    def test_recompressed_lhe_keeps_decoded_hash_but_updates_raw_hash(self):
        path = self.folder / 'events.lhe.gz'
        content = b'same event bytes'
        first = gzip.compress(content, mtime=1)
        second = gzip.compress(content, mtime=2)
        path.write_bytes(first)
        decoded = mc.lhe_sha256(path)
        raw = mc.sha256(path)
        path.write_bytes(second)
        self.assertEqual(mc.lhe_sha256(path), decoded)
        self.assertNotEqual(mc.sha256(path), raw)
        self.assertEqual(mc.sha256(path), hashlib.sha256(second).hexdigest())

    def test_changes_during_a_read_are_rejected_without_caching(self):
        path = self.folder / 'source'
        path.write_bytes(b'initial data')
        original_open = open

        class ChangingReader:
            def __init__(self):
                self.stream = original_open(path, 'rb')
                self.changed = False
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def fileno(self):
                return self.stream.fileno()
            def read(self, size):
                value = self.stream.read(size)
                if not self.changed:
                    path.write_bytes(b'changed data')
                    self.changed = True
                return value

        with patch.object(mc, 'open', create=True, side_effect=lambda *args: ChangingReader()):
            with self.assertRaisesRegex(mc.MatchingError, 'changed while hashing'):
                mc.sha256(path)
        self.assertEqual(len(mc._checksum_cache), 0)
        self.assertEqual(mc.sha256(path), hashlib.sha256(b'changed data').hexdigest())

    def test_path_replacement_during_a_read_is_rejected(self):
        path = self.folder / 'source'
        path.write_bytes(b'original content')
        original_read = mc._read_checksum

        def replace_before_open(path_arg, decompressed, signature):
            new = self.folder / 'new'
            new.write_bytes(b'replacement content')
            new.replace(path)
            return original_read(path_arg, decompressed, signature)

        with patch.object(mc, '_read_checksum', side_effect=replace_before_open):
            with self.assertRaisesRegex(mc.MatchingError, 'changed while hashing'):
                mc.sha256(path)
        self.assertEqual(len(mc._checksum_cache), 0)

    def test_cache_hits_also_check_for_a_concurrent_change(self):
        path = self.folder / 'source'
        path.write_bytes(b'initial content')
        mc.sha256(path)

        class ChangingCache(OrderedDict):
            def get(self, key, default=None):
                result = super().get(key, default)
                path.write_bytes(b'changed content')
                return result

        with patch.object(mc, '_checksum_cache', ChangingCache(mc._checksum_cache)):
            with self.assertRaisesRegex(mc.MatchingError, 'changed while hashing'):
                mc.sha256(path)

    def test_empty_files_are_not_cached(self):
        path = self.folder / 'empty'
        path.write_bytes(b'')
        self.assertEqual(mc.sha256(path), hashlib.sha256(b'').hexdigest())
        self.assertEqual(len(mc._checksum_cache), 0)

    def test_cache_is_bounded_and_eviction_keeps_correct_hashes(self):
        paths = [self.folder / ('source_%d' % i) for i in range(3)]
        for i, path in enumerate(paths):
            path.write_bytes(('content %d' % i).encode())
        with patch.object(mc, '_CHECKSUM_CACHE_LIMIT', 2), \
                patch.object(mc, '_read_checksum', wraps=mc._read_checksum) as reads:
            for path in paths:
                mc.sha256(path)
            self.assertEqual(len(mc._checksum_cache), 2)
            self.assertEqual(mc.sha256(paths[0]), hashlib.sha256(paths[0].read_bytes()).hexdigest())
            self.assertEqual(reads.call_count, 4)
            self.assertEqual(len(mc._checksum_cache), 2)

    def test_parallel_lookups_keep_consistent_digests(self):
        paths = [self.folder / ('source_%d' % i) for i in range(3)]
        for i, path in enumerate(paths):
            path.write_bytes(bytes([i]) * 32768)
        requests = paths * 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(mc.sha256, requests))
        self.assertEqual(values, [hashlib.sha256(path.read_bytes()).hexdigest() for path in requests])
        self.assertEqual(len(mc._checksum_cache), 3)

    def test_deleted_or_missing_files_cannot_use_a_cached_digest(self):
        path = self.folder / 'source'
        path.write_bytes(b'content')
        mc.sha256(path)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            mc.sha256(path)


if __name__ == '__main__':
    unittest.main()
