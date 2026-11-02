#!/usr/bin/env python3
"""Pad the string table of a Mach-O library to the 8-byte alignment macOS 27's
loader requires ("mis-aligned LINKEDIT string pool"), then sign it ad hoc.

The linker of Command Line Tools 27.2 leaves the string table after an odd
number of indirect symbols, i.e. at an offset that is 4 modulo 8. Usage:
fix_macho_alignment.py LIB [LIB...]
"""

import struct
import subprocess
import sys

MH_MAGIC_64 = 0xFEEDFACF
LC_SEGMENT_64, LC_SYMTAB, LC_CODE_SIGNATURE = 0x19, 0x2, 0x1D


def fix(path):
    data = bytearray(open(path, "rb").read())
    magic, _, _, _, ncmds, _, _, _ = struct.unpack_from("<8I", data, 0)
    if magic != MH_MAGIC_64:
        raise SystemExit(f"{path}: not a 64-bit little-endian Mach-O file")
    subprocess.run(["codesign", "--remove-signature", path], check=False,
                   capture_output=True)
    data = bytearray(open(path, "rb").read())
    offset, symtab, linkedit = 32, None, None
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, offset)
        if cmd == LC_SYMTAB:
            symtab = offset
        elif cmd == LC_SEGMENT_64 and data[offset + 8:offset + 24].rstrip(b"\0") == b"__LINKEDIT":
            linkedit = offset
        offset += size
    stroff = struct.unpack_from("<I", data, symtab + 16)[0]
    if stroff % 8 == 0:
        print(f"{path}: already aligned")
        return
    pad = 8 - stroff % 8
    data[stroff:stroff] = bytes(pad)
    struct.pack_into("<I", data, symtab + 16, stroff + pad)
    # __LINKEDIT: vmsize (offset 32), fileoff (40), filesize (48)
    vmsize, fileoff, filesize = struct.unpack_from("<QQQ", data, linkedit + 32)
    filesize += pad
    vmsize = max(vmsize, (filesize + 0x3FFF) & ~0x3FFF)
    struct.pack_into("<Q", data, linkedit + 32, vmsize)
    struct.pack_into("<Q", data, linkedit + 48, filesize)
    open(path, "wb").write(data)
    subprocess.run(["codesign", "--force", "--sign", "-", path], check=True,
                   capture_output=True)
    print(f"{path}: string table moved by {pad} bytes")


if __name__ == "__main__":
    for library in sys.argv[1:]:
        fix(library)
