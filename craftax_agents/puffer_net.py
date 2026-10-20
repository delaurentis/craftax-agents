"""PufferLib 5.0's Craftax policy, in NumPy: the network of ocean/craftax/craftax_net.h and src/puffercpu.c, its input packing
(tests/craftax_parity.py pack_symbolic_obs) and its action mask (ocean/craftax/craftax.h compute_action_mask). PufferLib is MIT
licensed (github.com/PufferAI/PufferLib, read at commit ea7f989, 2026-10-14).

    net = PufferNet.load("weights/craftax_1024x4.bin")           # hidden 1024, 4 MinGRU layers
    logits, value, h = net.step(pack(symbolic_obs), h)           # one step; h = net.zero_state(batch)
    a = masked_argmax(logits, action_mask(state))

Weights (flat float32, each tensor padded to a multiple of 8 floats, no biases):
    embed [154, 16] · proj [1024, 1635] · decoder [44, 1024] (43 logits + value) · 4 × MinGRU proj [3 × 1024, 1024]
Input: the 9 × 11 view as 8 ids a tile (block, item + 1, visible, then type + 1 of melee / passive / ranged / mob projectile /
player projectile, 0 when none), each looked up in its own slice of the 154-row embedding and summed → 99 × 16, then Craftax's
51 inventory and stat floats as they are (1,635).
"""
import numpy as np

OBS_ROWS, OBS_COLS, CHANNELS, EMB, VOCAB = 9, 11, 8, 16, 154
OFFSETS = np.array([0, 64, 72, 74, 90, 106, 122, 138])
N_BLOCKS, N_ITEMS, N_MOB_CLASSES, N_MOB_TYPES, N_SCALARS, N_ACTIONS = 37, 5, 5, 8, 51, 43
TILE = N_BLOCKS + N_ITEMS + N_MOB_CLASSES * N_MOB_TYPES + 1
SYMBOLIC = OBS_ROWS * OBS_COLS * TILE + N_SCALARS                                 # 8,268


def pack(obs):
    """Craftax-Symbolic's 8,268 floats [..., 8268] → the policy's 9 × 11 × 8 ids + 51 floats [..., 843]."""
    obs = np.asarray(obs, np.float32)
    lead = obs.shape[:-1]
    t = obs[..., :OBS_ROWS * OBS_COLS * TILE].reshape(lead + (OBS_ROWS, OBS_COLS, TILE))
    blocks, items = t[..., :N_BLOCKS], t[..., N_BLOCKS:N_BLOCKS + N_ITEMS]
    mobs = t[..., N_BLOCKS + N_ITEMS:-1].reshape(lead + (OBS_ROWS, OBS_COLS, N_MOB_CLASSES, N_MOB_TYPES))
    vis = t[..., -1]
    p = np.zeros(lead + (OBS_ROWS, OBS_COLS, CHANNELS), np.float32)
    p[..., 0] = blocks.argmax(-1) * vis
    p[..., 1] = (items.argmax(-1) + 1) * vis
    p[..., 2] = vis
    p[..., 3:] = (mobs.argmax(-1) + 1) * mobs.max(-1)
    return np.concatenate([p.reshape(lead + (-1,)), obs[..., OBS_ROWS * OBS_COLS * TILE:]], -1)


def action_mask(s):
    """craftax.h compute_action_mask on a JAX EnvState (one world, numpy leaves) → bool [43]. Craftax's action numbering."""
    inv = s.inventory
    g = lambda x: int(np.asarray(x))
    m = np.zeros(N_ACTIONS, bool)
    m[0] = True                                                                   # NOOP
    if g(s.is_sleeping) or g(s.is_resting):
        return m
    wood, stone, coal, iron, dia = g(inv.wood), g(inv.stone), g(inv.coal), g(inv.iron), g(inv.diamond)
    pick, sword, arrows, torches = g(inv.pickaxe), g(inv.sword), g(inv.arrows), g(inv.torches)
    arm = np.asarray(inv.armour).astype(int)
    m[1:6] = True                                                                 # LEFT RIGHT UP DOWN DO
    m[6] = g(s.player_energy) < 7 + 2 * g(s.player_dexterity)                     # SLEEP
    m[17] = float(np.asarray(s.player_health)) < 8 + g(s.player_strength)        # REST
    m[7] = m[9] = stone > 0                                                       # PLACE_STONE, PLACE_FURNACE
    m[8] = wood >= 2                                                              # PLACE_TABLE
    m[10] = g(inv.sapling) > 0                                                    # PLACE_PLANT
    m[28] = torches > 0                                                           # PLACE_TORCH
    m[11] = wood > 0 and pick < 1
    m[12] = wood > 0 and stone > 0 and pick < 2
    m[13] = wood > 0 and stone > 0 and iron > 0 and coal > 0 and pick < 3
    m[20] = wood > 0 and dia >= 3 and pick < 4
    m[14] = wood > 0 and sword < 1
    m[15] = wood > 0 and stone > 0 and sword < 2
    m[16] = wood > 0 and stone > 0 and iron > 0 and coal > 0 and sword < 3
    m[21] = wood > 0 and dia >= 2 and sword < 4
    m[25] = wood > 0 and stone > 0 and arrows < 99                                # MAKE_ARROW
    m[38] = wood > 0 and coal > 0 and torches < 99                                # MAKE_TORCH
    m[22] = (arm < 1).any() and iron >= 3 and coal >= 3                           # MAKE_IRON_ARMOUR
    m[23] = (arm < 2).any() and dia >= 3                                          # MAKE_DIAMOND_ARMOUR
    lv = g(s.player_level); r, c = (int(v) for v in np.asarray(s.player_position))
    item = int(np.asarray(s.item_map)[lv, r, c])
    from craftax.craftax.constants import ItemType
    m[18] = item == ItemType.LADDER_DOWN.value and g(np.asarray(s.monsters_killed)[lv]) >= 8 and lv < 8   # DESCEND
    m[19] = item == ItemType.LADDER_UP.value and lv > 0                           # ASCEND
    m[24] = g(inv.bow) > 0 and arrows > 0                                         # SHOOT_ARROW
    spells, mana = np.asarray(s.learned_spells), g(s.player_mana)
    m[26] = bool(spells[0]) and mana >= 2                                         # CAST_FIREBALL
    m[27] = bool(spells[1]) and mana >= 2                                         # CAST_ICEBALL
    m[29:35] = np.asarray(inv.potions) > 0                                        # DRINK_POTION_*
    m[35] = g(inv.books) > 0                                                      # READ_BOOK
    ench = mana >= 9 and (g(inv.ruby) > 0 or g(inv.sapphire) > 0)
    m[36] = ench and sword > 0                                                    # ENCHANT_SWORD
    m[37] = ench and arm.sum() > 0                                                # ENCHANT_ARMOUR
    m[42] = ench and g(inv.bow) > 0                                               # ENCHANT_BOW
    xp = g(s.player_xp)
    m[39] = xp > 0 and g(s.player_dexterity) < 5
    m[40] = xp > 0 and g(s.player_strength) < 5
    m[41] = xp > 0 and g(s.player_intelligence) < 5
    return m


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class PufferNet:
    def __init__(self, w, hidden=1024, layers=4):
        self.hidden, self.layers = hidden, layers
        cat = OBS_ROWS * OBS_COLS * EMB + N_SCALARS
        i = 0
        def take(n, shape):
            nonlocal i
            t = w[i:i + n].reshape(shape); i = (i + n + 7) & ~7
            return t
        self.embed = take(VOCAB * EMB, (VOCAB, EMB))
        self.proj = take(hidden * cat, (hidden, cat))
        self.decoder = take((N_ACTIONS + 1) * hidden, (N_ACTIONS + 1, hidden))
        self.gru = [take(3 * hidden * hidden, (3 * hidden, hidden)) for _ in range(layers)]
        assert i == len(w), f"weights: used {i} of {len(w)} floats — wrong hidden/layers?"

    @classmethod
    def load(cls, path, hidden=1024, layers=4):
        return cls(np.fromfile(path, np.float32), hidden, layers)

    def zero_state(self, batch=1):
        return np.zeros((self.layers, batch, self.hidden), np.float32)

    def step(self, x, h):
        """x [B, 843] packed input, h [layers, B, hidden] → logits [B, 43], value [B], new h."""
        x = np.asarray(x, np.float32)
        B = x.shape[0]
        ids = x[:, :OBS_ROWS * OBS_COLS * CHANNELS].reshape(B, OBS_ROWS * OBS_COLS, CHANNELS).astype(np.int64)
        rows = np.clip(OFFSETS + ids, OFFSETS, VOCAB - 1)                         # craftax_encode's clamps
        e = self.embed[rows].sum(2).reshape(B, -1)                                # [B, 99 × 16]
        z = np.concatenate([e, x[:, OBS_ROWS * OBS_COLS * CHANNELS:]], 1) @ self.proj.T
        h = h.copy()
        for l, W in enumerate(self.gru):                                          # puffercpu.c mingru(): gate + highway
            c = z @ W.T
            hid, gate, hw = c[:, :self.hidden], c[:, self.hidden:2 * self.hidden], c[:, 2 * self.hidden:]
            h_t = np.where(hid >= 0, hid + 0.5, _sigmoid(hid))
            out = h[l] + _sigmoid(gate) * (h_t - h[l])
            s = _sigmoid(hw)
            h[l] = out
            z = s * out + (1 - s) * z
        d = z @ self.decoder.T
        return d[:, :N_ACTIONS], d[:, N_ACTIONS], h


def masked_softmax(logits, mask):
    z = np.where(mask, logits, -np.inf)
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)
