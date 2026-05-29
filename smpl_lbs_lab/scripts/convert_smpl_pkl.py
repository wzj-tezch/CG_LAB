"""Convert legacy chumpy-based SMPL pickle to smplx-compatible Python 3 pickle."""

from __future__ import annotations

import copyreg
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp


class _ChStub:
    """Stand-in for chumpy.ch.Ch that only keeps the underlying numpy values."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            if "x" in state:
                self._data = np.asarray(state["x"])
            elif "dterms" in state and state["dterms"]:
                self._data = np.asarray(next(iter(state["dterms"].values())))
            else:
                for v in state.values():
                    if isinstance(v, np.ndarray):
                        self._data = np.asarray(v)
                        break
                else:
                    self._data = np.zeros(0)
        elif isinstance(state, tuple) and state:
            self._data = np.asarray(state[0])
        else:
            self._data = np.asarray(state)

    @property
    def r(self):
        return self._data


def _install_chumpy_stub() -> None:
    if "chumpy" in sys.modules:
        return
    chumpy = types.ModuleType("chumpy")
    ch = types.ModuleType("chumpy.ch")
    ch.Ch = _ChStub
    chumpy.ch = ch
    chumpy.ch_ops = types.ModuleType("chumpy.ch_ops")
    chumpy.ch_random = types.ModuleType("chumpy.ch_random")
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = ch
    sys.modules["chumpy.ch_ops"] = chumpy.ch_ops
    sys.modules["chumpy.ch_random"] = chumpy.ch_random


class _CompatUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
        remap = {
            "copy_reg": "copyreg",
            "__builtin__": "builtins",
            "numpy.core.multiarray": "numpy.core.multiarray",
            "numpy.core.numeric": "numpy",
        }
        module = remap.get(module, module)
        if module.startswith("chumpy"):
            return _ChStub
        if module == "scipy.sparse.csc" and name == "csc_matrix":
            return sp.csc_matrix
        return super().find_class(module, name)


def _to_native(obj):
    if sp.issparse(obj):
        return obj.tocsc()
    if isinstance(obj, np.ndarray):
        return np.asarray(obj)
    if isinstance(obj, _ChStub):
        return np.asarray(obj.r)
    if hasattr(obj, "r") and not isinstance(obj, np.ndarray):
        return np.asarray(obj.r)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_native(v) for v in obj)
    return obj


def convert_smpl_pkl(src: Path, dst: Path) -> Path:
    _install_chumpy_stub()
    with open(src, "rb") as f:
        data = _CompatUnpickler(f, encoding="latin1").load()
    out = _to_native(data)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dst


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path, nargs="?", default=None)
    args = parser.parse_args()
    dst = args.dst or args.src
    path = convert_smpl_pkl(args.src, dst)
    print("converted ->", path, path.stat().st_size)
