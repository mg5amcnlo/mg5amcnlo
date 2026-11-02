import subprocess
import tarfile

import numpy as np
import pytest

import madspace as ms

torch = pytest.importorskip("torch")


def make_context(tmp_path):
    ctx = ms.Context(device=ms.cpu_device(), thread_count=1)
    rng = np.random.default_rng(1)
    values = {
        "a.float": rng.normal(size=(1, 4, 50)),
        "b.float": rng.normal(size=(1, 3, 5)),
        "c.int": rng.integers(-100, 100, size=(1, 2, 7)).astype(np.int32),
        "d.scalar": rng.normal(size=(1, 1)),
    }
    for name, value in values.items():
        dtype = ms.DataType.int32 if value.dtype == np.int32 else ms.DataType.float64
        glob = ctx.define_global(name, dtype, list(value.shape[1:]))
        glob.torch().copy_(torch.from_numpy(value))
    return ctx, values


def test_roundtrip(tmp_path):
    ctx, values = make_context(tmp_path)
    path = str(tmp_path / "globals.tar")
    ctx.save_globals(path)

    ctx2 = ms.Context(device=ms.cpu_device(), thread_count=1)
    ctx2.load_globals(path)
    assert sorted(ctx2.global_names()) == sorted(values)
    for name, value in values.items():
        np.testing.assert_array_equal(
            ctx2.get_global(name).numpy(), ctx.get_global(name).numpy()
        )


def test_standard_tar(tmp_path):
    ctx, values = make_context(tmp_path)
    path = tmp_path / "globals.tar"
    ctx.save_globals(str(path))

    # system tar and the Python standard library can read it
    listing = subprocess.run(
        ["tar", "tf", str(path)], capture_output=True, text=True, check=True
    ).stdout.split()
    assert sorted(listing) == sorted(name + ".npy" for name in values)
    with tarfile.open(path) as tar:
        assert tar.getnames() == sorted(name + ".npy" for name in values)
        for name in values:
            tar.extract(name + ".npy", tmp_path / "out", filter="data")

    # and so can numpy, with the same values as the context holds
    for name in values:
        expected = ctx.get_global(name).numpy()
        loaded = np.load(tmp_path / "out" / (name + ".npy"))
        np.testing.assert_array_equal(loaded, expected)


def test_extract_with_system_tar(tmp_path):
    ctx, values = make_context(tmp_path)
    path = tmp_path / "globals.tar"
    ctx.save_globals(str(path))
    out = tmp_path / "out"
    out.mkdir()
    subprocess.run(["tar", "xf", str(path), "-C", str(out)], check=True)
    assert sorted(p.name for p in out.iterdir()) == sorted(
        name + ".npy" for name in values
    )


def anonymous(info):
    # the uid of a cluster account can overflow the tar header fields
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def test_read_tar_from_other_tool(tmp_path):
    ctx, values = make_context(tmp_path)
    first = tmp_path / "first.tar"
    ctx.save_globals(str(first))
    src = tmp_path / "src"
    src.mkdir()
    with tarfile.open(first) as tar:
        tar.extractall(src, filter="data")

    # an archive written by Python's tarfile (GNU format) and by system tar
    for fmt, name in [
        (tarfile.GNU_FORMAT, "gnu.tar"),
        (tarfile.USTAR_FORMAT, "ustar.tar"),
    ]:
        other = tmp_path / name
        with tarfile.open(other, "w", format=fmt) as tar:
            for p in sorted(src.iterdir()):
                tar.add(p, arcname=p.name, filter=anonymous)
        ctx2 = ms.Context(device=ms.cpu_device(), thread_count=1)
        ctx2.load_globals(str(other))
        for gname in values:
            np.testing.assert_array_equal(
                ctx2.get_global(gname).numpy(), ctx.get_global(gname).numpy()
            )


def test_missing_file(tmp_path):
    ctx = ms.Context(device=ms.cpu_device(), thread_count=1)
    with pytest.raises(RuntimeError):
        ctx.load_globals(str(tmp_path / "nonexistent.tar"))
