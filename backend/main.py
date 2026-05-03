import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math
import google.generativeai as genai

# =========================
# GLOBAL STATE CONTRACT
# =========================

MINE = -9
HIDDEN = -1
FLAGGED = -2

# =========================
# FASTAPI SETUP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GEMINI SETUP
# =========================

GEMINI_API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=GEMINI_API_KEY)

current_field = None

# =========================
# MODEL LOAD
# =========================

try:
    model = tf.keras.models.load_model("hazard_model.keras")
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model not found or failed to load: {e}")
    model = None

# ============================================================
#                    CORE ENCODING (MATCH TRAINER)
# ============================================================

def compute_frontier(board):
    size = board.shape[0]
    frontier = np.zeros((size, size), dtype=np.float32)

    for r in range(size):
        for c in range(size):
            if board[r, c] != HIDDEN:
                continue

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        if board[nr, nc] >= 0:
                            frontier[r, c] = 1
                            break
                if frontier[r, c] == 1:
                    break

    return frontier


def encode_board(board):
    size = int(math.sqrt(len(board)))
    board = np.array(board).reshape(size, size)

    tensor = np.zeros((1, size, size, 11), dtype=np.float32)

    for r in range(size):
        for c in range(size):
            val = board[r, c]

            if val == HIDDEN or val == FLAGGED or val is None or val == "X":
                tensor[0, r, c, 0] = 1
            elif isinstance(val, (int, np.integer)) and 0 <= val <= 8:
                tensor[0, r, c, val + 1] = 1

    tensor[0, :, :, 10] = compute_frontier(board)
    return tensor

# ============================================================
#                INPUT NORMALIZATION (CRITICAL FIX)
# ============================================================

def normalize_grid(grid):
    return [
        HIDDEN if x is None or x == "X" else x
        for x in grid
    ]

# ============================================================
#                      DETECTION ENGINE
# ============================================================

def get_deterministic_moves(visible_board, size):
    safe_moves = set()
    known_hazards = set()
    clue_equations = []

    for r in range(size):
        for c in range(size):
            clue = visible_board[r, c]
            if clue <= 0:
                continue

            unknowns = set()
            hazards_found = 0

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc

                    if 0 <= nr < size and 0 <= nc < size:
                        if visible_board[nr, nc] == HIDDEN:
                            unknowns.add((nr, nc))
                        elif visible_board[nr, nc] == FLAGGED:
                            hazards_found += 1

            if not unknowns:
                continue

            remaining = clue - hazards_found
            clue_equations.append((remaining, unknowns))

            if remaining == len(unknowns):
                known_hazards.update(unknowns)
            if remaining == 0:
                safe_moves.update(unknowns)

    return list(safe_moves), list(known_hazards)

# ============================================================
#                  RECURSIVE REVEAL (REAL GAME RULES)
# ============================================================

def reveal(clues, visible, r, c):
    size = clues.shape[0]

    if not (0 <= r < size and 0 <= c < size):
        return 0

    if visible[r, c] != HIDDEN:
        return 0

    if clues[r, c] == MINE:
        return 0

    visible[r, c] = clues[r, c]
    revealed = 1

    if clues[r, c] == 0:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                revealed += reveal(clues, visible, r + dr, c + dc)

    return revealed

# ============================================================
#                   FIELD GENERATION
# ============================================================

def generate_field(size, hazards):
    field = np.zeros((size, size), dtype=int)

    indices = np.random.choice(size * size, hazards, replace=False)
    field.ravel()[indices] = MINE

    clues = np.zeros((size, size), dtype=int)

    for r in range(size):
        for c in range(size):
            if field[r, c] == MINE:
                clues[r, c] = MINE
                continue

            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        if field[nr, nc] == MINE:
                            count += 1

            clues[r, c] = count

    return field, clues

# ============================================================
#                   GAME SIMULATION
# ============================================================

def simulate_single_game(size, hazards):
    field, clues = generate_field(size, hazards)

    visible = np.full((size, size), HIDDEN)
    safe_total = (size * size) - hazards
    cleared = 0
    flagged = 0

    # first safe click
    while True:
        r, c = np.random.randint(0, size), np.random.randint(0, size)
        if field[r, c] != MINE:
            cleared += reveal(clues, visible, r, c)
            break

    while cleared < safe_total:

        safe_moves, known_mines = get_deterministic_moves(visible, size)

        for hr, hc in known_mines:
            if visible[hr, hc] != FLAGGED:
                visible[hr, hc] = FLAGGED
                flagged += 1

        if safe_moves:
            r, c = safe_moves[0]

        else:
            if model is None:
                return False, visible.flatten().tolist(), {}

            encoded = encode_board(visible.flatten().tolist())
            preds = model.predict(encoded, verbose=0)[0].flatten()

            for i in range(size * size):
                rr, cc = divmod(i, size)
                if visible[rr, cc] != HIDDEN:
                    preds[i] = 999.0

            best = np.argmin(preds)
            r, c = divmod(best, size)

        if field[r, c] == MINE:
            flat = visible.flatten().tolist()
            flat[r * size + c] = "X"

            return False, flat, {
                "engine": "CNN / Logic Hybrid",
                "cleared_percent": round((cleared / safe_total) * 100, 1)
            }

        cleared += reveal(clues, visible, r, c)

    return True, visible.flatten().tolist(), {
        "engine": "Logic",
        "cleared_percent": 100.0
    }

# ============================================================
# REST OF API (UNCHANGED LOGICALLY)
# ============================================================

class BatchRequest(BaseModel):
    size: int
    hazards: int
    num_games: int

class ExplanationRequest(BaseModel):
    cleared_percent: float
    engine: str
    raw_explanation: str

@app.get("/")
def root():
    return {"status": "running"}
# ============================================================
#                   API NORMALIZATION FIXES
# ============================================================

@app.post("/analyze-risk")
async def analyze_risk(data: dict):
    global model

    grid = normalize_grid(data.get("grid"))

    size = int(math.sqrt(len(grid)))

    if model is None:
        return {
            "heatmap": [0.1] * (size * size),
            "cnn_confidence": 0,
            "bayesian_est": 15,
            "safest_tile_index": -1
        }

    encoded = encode_board(grid)

    preds = model.predict(encoded, verbose=0)[0].flatten()
    heatmap = preds.tolist()

    min_risk = float("inf")
    safest = -1

    for i, val in enumerate(grid):
        if val == HIDDEN:
            if heatmap[i] < min_risk:
                min_risk = heatmap[i]
                safest = i

    avg = float(np.mean(heatmap))

    return {
        "heatmap": heatmap,
        "cnn_confidence": round((1 - avg) * 100, 2),
        "bayesian_est": round(avg * 100, 2),
        "safest_tile_index": safest
    }

@app.post("/batch-solve")
async def batch_solve(req: BatchRequest):
    wins = []
    losses = []

    try:
        for _ in range(req.num_games):
            won, board, telemetry = simulate_single_game(req.size, req.hazards)

            if telemetry is None:
                telemetry = {
                    "engine": "unknown",
                    "cleared_percent": 0
                }

            entry = {
                "board": board,
                "telemetry": telemetry
            }

            if won:
                wins.append(entry)
            else:
                losses.append(entry)

        return {
            "win_rate": round((len(wins) / req.num_games) * 100, 2),
            "wins": wins,
            "losses": losses
        }

    except Exception as e:
        print("BATCH-SOLVE CRASHED:", str(e))
        return {
            "error": str(e),
            "win_rate": 0,
            "wins": [],
            "losses": []
        }

