"""The native craftax-wasm (libcraftax.dylib|.so) through ctypes, for the exporters' --check and check_model.py.

    g = Game(flags=4); g.reset(7)              # 4 = CX_ACTION_MASK; sleeps fast-forwarded as in training (and in the browser)
    obs, mask = g.obs(), g.mask()              # [843] float32 (a copy), [43] bool
    done = g.step(a); g.score()

Where the library is looked for, first found wins:
    $CRAFTAX_ENGINE_LIB                        the library file itself (or the folder holding it)
    engine/build/                              craftax-wasm as a git submodule of this kit: make -C engine native
    ../engine/build/                           the kit and the engine side by side (one checkout holding both)
"""
import ctypes, os, sys
import numpy as np

from craftax_agents import KIT

OBS, ACTIONS = 843, 43
NAMES = ("libcraftax.dylib", "libcraftax.so")
_cx = None


def find_lib():
    """The engine library's path, or None."""
    env = os.environ.get("CRAFTAX_ENGINE_LIB")
    if env:
        cands = [os.path.join(env, n) for n in NAMES] if os.path.isdir(env) else [env]
    else:
        cands = [os.path.join(d, "build", n) for d in (os.path.join(KIT, "engine"), os.path.join(os.path.dirname(KIT), "engine")) for n in NAMES]
    return next((p for p in cands if os.path.isfile(p)), None)


def lib():
    global _cx
    if _cx is None:
        path = find_lib()
        if path is None:
            sys.exit("no native craftax-wasm library: set CRAFTAX_ENGINE_LIB, or build it with `make -C engine native` "
                     "(git submodule update --init first; see README.md)")
        cx = ctypes.CDLL(path)
        cx.cx_new.restype = ctypes.c_void_p; cx.cx_obs.restype = ctypes.POINTER(ctypes.c_float)
        cx.cx_mask.restype = ctypes.POINTER(ctypes.c_uint8); cx.cx_score.restype = ctypes.c_float; cx.cx_reward.restype = ctypes.c_float
        for f in (cx.cx_reset, cx.cx_step, cx.cx_obs, cx.cx_mask, cx.cx_score, cx.cx_reward, cx.cx_done, cx.cx_free):
            f.argtypes = [ctypes.c_void_p] + ([ctypes.c_uint32] if f is cx.cx_reset else [ctypes.c_int] if f is cx.cx_step else [])
        cx.cx_new.argtypes = [ctypes.c_int]
        _cx = cx
    return _cx


class Game:
    def __init__(self, flags=4):
        self.cx = lib(); self.g = self.cx.cx_new(flags)
    def reset(self, seed): self.cx.cx_reset(self.g, seed)
    def step(self, a): return bool(self.cx.cx_step(self.g, int(a)))           # True when this step ended the episode
    def obs(self): return np.ctypeslib.as_array(self.cx.cx_obs(self.g), (OBS,)).copy()
    def mask(self): return np.ctypeslib.as_array(self.cx.cx_mask(self.g), (ACTIONS,)).astype(bool)
    def score(self): return float(self.cx.cx_score(self.g))
    def reward(self): return float(self.cx.cx_reward(self.g))
    def __del__(self):
        if getattr(self, "g", None): self.cx.cx_free(self.g); self.g = None
