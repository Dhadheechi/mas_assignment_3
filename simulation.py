"""
Multi-agent formation control spelling out "DHADHEECHI" using
consensus-based formation control with a pinned virtual leader.

Control law (matches Algorithm 6 in the MAS lecture slides):
    u_i = sum_j a_ij [(p_j - p_i) - (r_j - r_i)]   <- agree on shape
          - b_i [ p_i - r_i - c(t) ]                <- pin to anchor
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(7)

N_AGENTS = 20
WORD = "DHADHEECHI"
LETTER_SPACING = 3.2     # horizontal distance between successive letter anchors
HOLD_STEPS = 800         # simulation steps to let formation converge on each letter
DT = 0.05

# ---------------------------------------------------------------------
# 1. Hand-built letter skeletons (each letter = list of strokes,
#    each stroke = polyline of (x, y) points)
# ---------------------------------------------------------------------
def arc(cx, cy, r, a0, a1, n=12):
    """Helper: polyline approximating a circular arc, for curved strokes like C."""
    angles = np.linspace(a0, a1, n)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

SEGMENT_LETTERS = {
    "D": [
        [(0, 0), (0, 4)],
        [(0, 4), (1.2, 4), (2, 3), (2, 1), (1.2, 0), (0, 0)],
    ],
    "H": [
        [(0, 0), (0, 4)],
        [(2, 0), (2, 4)],
        [(0, 2), (2, 2)],
    ],
    "A": [
        [(0, 0), (1, 4), (2, 0)],
        [(0.5, 1.5), (1.5, 1.5)],
    ],
    "E": [
        [(0, 0), (0, 4)],
        [(0, 4), (1.8, 4)],
        [(0, 2), (1.8, 2)],
        [(0, 0), (1.8, 0)],
    ],
    "C": [
        arc(1.0, 2.0, 1.6, np.radians(40), np.radians(320)),
    ],
    "I": [
        [(-1, 4), (1, 4)],
        [(0, 4), (0, 0)],
        [(-1, 0), (1, 0)],
    ],
}


def sample_letter(letter, n_points=N_AGENTS):
    """Evenly (by arc length) sample n_points along a letter's strokes,
    return them centered at the origin."""
    strokes = SEGMENT_LETTERS[letter]
    segs, seg_lens = [], []
    for stroke in strokes:
        stroke = np.array(stroke, dtype=float)
        for a, b in zip(stroke[:-1], stroke[1:]):
            segs.append((a, b))
            seg_lens.append(np.linalg.norm(b - a))
    seg_lens = np.array(seg_lens)
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    sample_at = np.linspace(0, total_len, n_points, endpoint=False)
    sampled = np.zeros((n_points, 2))
    for k, s in enumerate(sample_at):
        idx = np.clip(np.searchsorted(cum_len, s) - 1, 0, len(segs) - 1)
        a, b = segs[idx]
        frac = (s - cum_len[idx]) / (seg_lens[idx] + 1e-9)
        sampled[k] = a + frac * (b - a)

    sampled -= sampled.mean(axis=0)
    return sampled


# ---------------------------------------------------------------------
# 2. Communication graph: connected Erdos-Renyi graph over 20 agents
# ---------------------------------------------------------------------
def connected_erdos_renyi(n, p=0.25, seed=0):
    g = nx.erdos_renyi_graph(n, p, seed=seed)
    trial = seed
    while not nx.is_connected(g):
        trial += 1
        g = nx.erdos_renyi_graph(n, p, seed=trial)
    return g

G = connected_erdos_renyi(N_AGENTS, p=0.22, seed=3)
A = nx.to_numpy_array(G)  # adjacency (weights a_ij = 1 for connected pairs)

# ---------------------------------------------------------------------
# 3. Build the sequence of offset targets r_i for each letter, with
#    Hungarian assignment between consecutive letters to avoid
#    unnecessary crossing paths.
# ---------------------------------------------------------------------
letters_raw = [sample_letter(ch) for ch in WORD]

letters_assigned = [letters_raw[0]]  # first letter: identity order
current_order = letters_raw[0]
for k in range(1, len(letters_raw)):
    target = letters_raw[k]
    # cost[i, j] = distance from agent i's current target point to
    # candidate point j of the next letter
    cost = np.linalg.norm(current_order[:, None, :] - target[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    reordered = target[col_ind]
    letters_assigned.append(reordered)
    current_order = reordered

# anchor c(t) for each letter: lay the word out left to right
anchors = [np.array([k * LETTER_SPACING, 0.0]) for k in range(len(WORD))]

# ---------------------------------------------------------------------
# 4. Pinned consensus formation controller, simulated with Euler steps
# ---------------------------------------------------------------------
p = rng.uniform(-2, 2, size=(N_AGENTS, 2)) + anchors[0]  # random initial positions near first anchor
pin_mask = np.zeros(N_AGENTS)
pin_mask[rng.choice(N_AGENTS, size=3, replace=False)] = 1.0  # only 3 agents see the anchor directly
b_gain = 0.6

all_frames = []  # list of (positions, anchor_index) for animation

for li, r in enumerate(letters_assigned):
    c = anchors[li]
    for _ in range(HOLD_STEPS):
        # formation term: sum_j a_ij [(p_j - p_i) - (r_j - r_i)]
        diff_p = p[None, :, :] - p[:, None, :]        # p_j - p_i
        diff_r = r[None, :, :] - r[:, None, :]        # r_j - r_i
        formation_err = (diff_p - diff_r) * A[:, :, None]
        u_form = formation_err.sum(axis=1)

        # pinning term: -b_i [p_i - r_i - c]
        u_pin = -b_gain * pin_mask[:, None] * (p - r - c[None, :])

        u = u_form + u_pin
        p = p + DT * u
        all_frames.append(p.copy())

# ---------------------------------------------------------------------
# 5. Animate
# ---------------------------------------------------------------------
all_frames_arr = np.array(all_frames)  # (n_steps, N_AGENTS, 2)
y_min, y_max = all_frames_arr[:, :, 1].min(), all_frames_arr[:, :, 1].max()
x_min, x_max = all_frames_arr[:, :, 0].min(), all_frames_arr[:, :, 0].max()
pad = 1.0

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(x_min - pad, x_max + pad)
ax.set_ylim(y_min - pad, y_max + pad)
ax.set_aspect("equal")
ax.axis("off")

scat = ax.scatter([], [], c="royalblue", s=45, zorder=5)
edge_lines = [ax.plot([], [], color="lightgray", lw=0.6, zorder=1)[0]
              for _ in G.edges()]
title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", fontsize=13)

STRIDE = 4  # skip frames for a reasonable file size / render time

def init():
    scat.set_offsets(np.zeros((N_AGENTS, 2)))
    return [scat, title] + edge_lines

def update(frame_idx):
    pts = all_frames[frame_idx * STRIDE]
    scat.set_offsets(pts)
    for line, (i, j) in zip(edge_lines, G.edges()):
        line.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
    letter_idx = (frame_idx * STRIDE) // HOLD_STEPS
    letter_idx = min(letter_idx, len(WORD) - 1)
    title.set_text(f"Forming: {WORD}  (current target letter: '{WORD[letter_idx]}')")
    return [scat, title] + edge_lines

n_frames = len(all_frames) // STRIDE
ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init,
                               interval=30, blit=True)

ani.save("/./dhadheechi_formation.gif", writer="pillow", fps=30, dpi=90)
print("saved animation")
