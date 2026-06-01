"""
BrainBlock Packing Environment
================================
Gymnasium-compatible environment for the 8x5 tetromino packing puzzle.

Board: W=8, H=5 (40 cells)
Pieces: 2×I, 2×O, 2×L, 2×Z, 2×T  (10 pieces × 4 cells = 40)
Action space: orientation (8) × x (8) × y (5)  →  flat index [0, 319]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

# ---------------------------------------------------------------------------
# Piece definitions  (cells relative to anchor (0,0))
# ---------------------------------------------------------------------------
# Each piece is a list of (dx, dy) offsets.  We precompute all 8 transforms.

_PIECE_CELLS = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "L": [(0, 0), (0, 1), (0, 2), (1, 2)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
}

PIECE_TYPES = ["I", "O", "L", "Z", "T"]
PIECE_TYPE_TO_IDX = {p: i for i, p in enumerate(PIECE_TYPES)}

# Inventory: 2 of each
INVENTORY = ["I", "I", "O", "O", "L", "L", "Z", "Z", "T", "T"]


def _rotate_90(cells):
    """Rotate cells 90° CCW: (x,y) → (-y, x)."""
    return [(-y, x) for x, y in cells]


def _reflect_x(cells):
    """Reflect about x-axis: (x,y) → (-x, y)."""
    return [(-x, y) for x, y in cells]


def _normalize(cells):
    """Shift cells so min x=0, min y=0, then sort."""
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    normed = tuple(sorted((x - min_x, y - min_y) for x, y in cells))
    return normed


def _all_transforms(cells):
    """Return list of 8 canonical cell-tuples (some may be duplicates for symmetric pieces)."""
    variants = []
    cur = cells
    for _ in range(4):
        variants.append(_normalize(cur))
        variants.append(_normalize(_reflect_x(cur)))
        cur = _rotate_90(cur)
    # Keep all 8 slots; duplicates are OK (spec allows it)
    result = []
    seen_unique = []
    for v in variants:
        result.append(v)
    return result  # always length 8


# Precompute transforms for every piece type
PIECE_TRANSFORMS: Dict[str, List[Tuple]] = {
    p: _all_transforms(_PIECE_CELLS[p]) for p in PIECE_TYPES
}

W, H = 8, 5
N_ORIENTATIONS = 8
N_ACTIONS = N_ORIENTATIONS * W * H  # 320


def decode_action(action: int) -> Tuple[int, int, int]:
    """Flat action index → (orientation, x, y)."""
    y = action % H
    x = (action // H) % W
    o = action // (H * W)
    return o, x, y


def encode_action(o: int, x: int, y: int) -> int:
    return o * H * W + x * H + y


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BrainBlockEnv(gym.Env):
    """
    BrainBlock 8×5 tetromino packing environment.

    Observation (flat float32 vector):
        - board state:        40 binary values (row-major, filled=1)
        - current piece:       5 one-hot values
        - remaining counts:    5 integer values (normalised to [0,1])

    Action: flat integer in [0, 319]
    """

    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(self, reward_fn: str = "sparse", render_mode: Optional[str] = None):
        super().__init__()
        assert reward_fn in ("sparse", "dense"), "reward_fn must be 'sparse' or 'dense'"
        self.reward_fn = reward_fn
        self.render_mode = render_mode

        # Spaces
        # Observation: 40 (board) + 5 (current piece one-hot) + 5 (remaining)
        obs_dim = W * H + len(PIECE_TYPES) + len(PIECE_TYPES)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Internal state (initialised in reset)
        self.board: Optional[np.ndarray] = None
        self.queue: Optional[List[str]] = None
        self.step_count: int = 0
        self.invalid_action_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.board = np.zeros((H, W), dtype=np.int8)
        self.queue = list(INVENTORY)
        self.np_random.shuffle(self.queue)
        self.step_count = 0
        self.invalid_action_count = 0
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.board is not None, "Call reset() first."
        o, x, y = decode_action(action)
        piece_type = self.queue[0]
        cells = PIECE_TRANSFORMS[piece_type][o]

        legal = self._is_legal(cells, x, y)

        if legal:
            self._place(cells, x, y)
            self.queue.pop(0)
            self.step_count += 1

            terminated = len(self.queue) == 0
            reward = self._compute_reward(legal=True, cells=cells, x=x, y=y,
                                          terminated=terminated)
            truncated = False
        else:
            self.invalid_action_count += 1
            terminated = True          # hard termination on invalid action
            truncated = False
            reward = self._compute_reward(legal=False, cells=None, x=x, y=y,
                                          terminated=False)

        obs = self._get_obs()
        info = {
            "solved": terminated and legal and len(self.queue) == 0,
            "invalid": not legal,
            "step": self.step_count,
            "covered": int(self.board.sum()),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_legal(self, cells, ax, ay) -> bool:
        for dx, dy in cells:
            nx, ny = ax + dx, ay + dy
            if not (0 <= nx < W and 0 <= ny < H):
                return False
            if self.board[ny, nx] != 0:
                return False
        return True

    def _place(self, cells, ax, ay):
        for dx, dy in cells:
            self.board[ay + dy, ax + dx] = 1

    def _get_obs(self) -> np.ndarray:
        board_flat = self.board.flatten().astype(np.float32)

        # Current piece one-hot
        piece_oh = np.zeros(len(PIECE_TYPES), dtype=np.float32)
        if self.queue:
            piece_oh[PIECE_TYPE_TO_IDX[self.queue[0]]] = 1.0

        # Remaining counts (normalised by max 2)
        remaining = np.zeros(len(PIECE_TYPES), dtype=np.float32)
        for p in self.queue:
            remaining[PIECE_TYPE_TO_IDX[p]] += 1
        remaining /= 2.0

        return np.concatenate([board_flat, piece_oh, remaining])

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _compute_reward(self, legal: bool, cells, x: int, y: int,
                        terminated: bool) -> float:
        if self.reward_fn == "sparse":
            return self._reward_sparse(legal, terminated)
        else:
            return self._reward_dense(legal, cells, x, y, terminated)

    def _reward_sparse(self, legal: bool, terminated: bool) -> float:
        """
        Sparse reward:
            +10   on full board completion
             -1   on invalid action
              0   otherwise
        """
        if not legal:
            return -1.0
        if terminated:
            return 10.0
        return 0.0

    def _reward_dense(self, legal: bool, cells, x: int, y: int,
                      terminated: bool) -> float:
        """
        Dense reward:
            +0.4  per cell placed (max +1.6 per step)
            +0.2  bonus if the placed piece is adjacent to existing filled cells
                  (encourages compact packing)
            +10   on completion
            -1    on invalid action
        """
        if not legal:
            return -1.0

        reward = 0.4 * len(cells)  # cells-placed reward

        # Adjacency bonus: encourage filling contiguously
        adj_bonus = 0.0
        for dx, dy in cells:
            nx, ny = x + dx, y + dy
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nnx, nny = nx + ddx, ny + ddy
                if 0 <= nnx < W and 0 <= nny < H and self.board[nny, nnx] == 1:
                    adj_bonus = 0.2  # just a flag, not per-cell
                    break

        reward += adj_bonus

        if terminated:
            reward += 10.0

        return reward

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_valid_action_mask(self) -> np.ndarray:
        """Return boolean mask of shape (320,) — True if action is legal."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if not self.queue:
            return mask
        piece_type = self.queue[0]
        for o in range(N_ORIENTATIONS):
            cells = PIECE_TRANSFORMS[piece_type][o]
            for x in range(W):
                for y in range(H):
                    if self._is_legal(cells, x, y):
                        mask[encode_action(o, x, y)] = True
        return mask

    def _render_ansi(self) -> str:
        piece_char = self.queue[0] if self.queue else "?"
        lines = [f"Step {self.step_count} | Next: {piece_char} | Queue: {self.queue}"]
        lines.append("  " + " ".join(str(i) for i in range(W)))
        for r in range(H):
            row = " ".join("█" if self.board[r, c] else "·" for c in range(W))
            lines.append(f"{r} {row}")
        return "\n".join(lines)
