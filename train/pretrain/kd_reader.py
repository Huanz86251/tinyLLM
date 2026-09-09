# kd_reader.py
import os, json, lmdb, struct, numpy as np, xxhash

MAGIC = b"KDTOPK01"
HDR   = struct.Struct("<8sIHHII")

def _xxh3_128_ids(ids_list):
    arr = np.asarray(ids_list, dtype="<u4", order="C")
    h = xxhash.xxh3_128(); h.update(arr.tobytes())
    dig = h.intdigest()
    lo = int(np.uint64(dig & ((1<<64)-1))); hi = int(np.uint64(dig >> 64))
    return struct.pack("<QQ", hi, lo)

def _decode(blob: bytes):
    magic, T, K, flags, lidx, lval = HDR.unpack_from(blob, 0)
    assert magic == MAGIC, "bad magic"
    off = HDR.size
    idx = np.frombuffer(memoryview(blob)[off:off+lidx], dtype=np.uint32).reshape(T, K); off += lidx
    val = np.frombuffer(memoryview(blob)[off:off+lval], dtype=np.float16).reshape(T, K)
    return idx, val

class KDFetcher:
    def __init__(self, out_dir: str):
        mani = json.load(open(os.path.join(out_dir, "manifest.json"), "r", encoding="utf-8"))
        self.envs = []
        for rel in mani["shards"]:
            path = os.path.join(out_dir, rel)
            self.envs.append(lmdb.open(path, readonly=True, lock=False, readahead=False, max_readers=4096))
    def close(self):
        for e in self.envs: e.close()
    def get(self, input_ids) -> tuple[np.ndarray, np.ndarray] | None:
        key = _xxh3_128_ids(input_ids)
        # 依次查分片，命中即返
        for e in self.envs:
            with e.begin(write=False) as txn:
                blob = txn.get(key)
            if blob is not None:
                return _decode(blob)
        return None
