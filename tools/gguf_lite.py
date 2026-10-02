"""Minimal GGUF v3 reader/writer for ds4's repack tools: metadata and tensor infos are parsed, tensor data
is copied by offset (never interpreted)."""
import hashlib
import os
import struct

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 8: (32, 34), 10: (256, 84), 11: (256, 110),
         12: (256, 144),
         13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 20: (32, 18),
         21: (256, 110), 22: (256, 82), 23: (256, 136), 30: (1, 2), 39: (32, 17), 42: (64, 18)}
CHUNK = 64 << 20


def nbytes(ttype, dims):
    if ttype not in BLOCK:
        raise ValueError("unknown tensor type %d" % ttype)
    elems, size = BLOCK[ttype]
    if dims[0] % elems:
        raise ValueError("row of %d is not a whole number of %d-blocks" % (dims[0], elems))
    n = dims[0] // elems * size
    for d in dims[1:]:
        n *= d
    return n


def _align(n, a):
    return (n + a - 1) // a * a


class Reader:
    def __init__(self, path):
        self.path = str(path)
        self.size = os.path.getsize(self.path)
        with open(self.path, "rb") as f:
            self._f = f
            if f.read(4) != b"GGUF":
                raise ValueError("%s: not a GGUF file" % self.path)
            self.version = self._u("I")
            n_t, n_kv = self._u("Q"), self._u("Q")
            self.kv = {}
            for _ in range(n_kv):
                key = self._str()
                vt = self._u("I")
                self.kv[key] = (vt, self._val(vt))
            self.tensors = []
            for _ in range(n_t):
                name = self._str()
                dims = [self._u("Q") for _ in range(self._u("I"))]
                ttype, off = self._u("I"), self._u("Q")
                self.tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
            self.alignment = self.kv.get("general.alignment", (T_U32, 32))[1]
            self.data_start = _align(f.tell(), self.alignment)
        for t in self.tensors:
            t["nbytes"] = nbytes(t["type"], t["dims"])
            t["abs"] = self.data_start + t["offset"]
            if t["abs"] + t["nbytes"] > self.size:
                raise ValueError("%s: tensor %s ends past the file (%d > %d)" % (
                    self.path, t["name"], t["abs"] + t["nbytes"], self.size))

    def _u(self, fmt):
        return struct.unpack("<" + fmt, self._f.read(struct.calcsize("<" + fmt)))[0]

    def _str(self):
        return self._f.read(self._u("Q")).decode("utf-8")

    def _val(self, vt):
        if vt in _SCALAR:
            return self._u(_SCALAR[vt])
        if vt == T_STR:
            return self._str()
        if vt == T_ARR:
            et, n = self._u("I"), self._u("Q")
            return (et, [self._val(et) for _ in range(n)])
        raise ValueError("unknown metadata type %d" % vt)


def _pack_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_val(vt, v):
    if vt in _SCALAR:
        return struct.pack("<" + _SCALAR[vt], v)
    if vt == T_STR:
        return _pack_str(v)
    if vt == T_ARR:
        et, items = v
        return struct.pack("<IQ", et, len(items)) + b"".join(_pack_val(et, x) for x in items)
    raise ValueError("unknown metadata type %d" % vt)


def write(path, kv, tensors, alignment):
    """Writes a GGUF v3 file; returns {tensor name: sha256 hex of its data}."""
    head = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(kv))]
    for key, (vt, v) in kv.items():
        head += [_pack_str(key), struct.pack("<I", vt), _pack_val(vt, v)]
    off = 0
    for t in tensors:
        t["offset"] = off
        head += [_pack_str(t["name"]), struct.pack("<I", len(t["dims"])),
                 b"".join(struct.pack("<Q", d) for d in t["dims"]), struct.pack("<IQ", t["type"], off)]
        off = _align(off + t["nbytes"], alignment)
    header = b"".join(head)
    shas = {}
    with open(path, "wb") as out:
        out.write(header + b"\0" * (_align(len(header), alignment) - len(header)))
        for t in tensors:
            h = hashlib.sha256()
            if "data" in t:
                if len(t["data"]) != t["nbytes"]:
                    raise ValueError("%s: %d bytes of data for %d" % (t["name"], len(t["data"]), t["nbytes"]))
                out.write(t["data"])
                h.update(t["data"])
            else:
                src, at = t["src"]
                left = t["nbytes"]
                with open(src, "rb") as f:
                    f.seek(at)
                    while left:
                        buf = f.read(min(CHUNK, left))
                        if not buf:
                            raise ValueError("%s: source %s ends early" % (t["name"], src))
                        out.write(buf)
                        h.update(buf)
                        left -= len(buf)
            shas[t["name"]] = h.hexdigest()
            pad = _align(t["nbytes"], alignment) - t["nbytes"]
            out.write(b"\0" * pad)
    return shas
