import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math
import google.generativeai as genai
import os

# Setup FastAPI app
app = FastAPI()

# Enable CORS for React frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GEMINI AI SETUP ---
# Replace this with your actual key from aistudio.google.com
GEMINI_API_KEY = "YOUR_API_KEY_HERE" 
genai.configure(api_key=GEMINI_API_KEY)


current_field = None 

# Load the trained CNN brain
try:
    model = tf.keras.models.load_model('hazard_model.h5')
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model not found or failed to load: {e}")
    model = None

# --- HELPER LOGIC ---

def encode_board(board):
    size = int(math.sqrt(len(board)))
    tensor = np.zeros((1, size, size, 10))
    for r in range(size):
        for c in range(size):
            val = board[r * size + c] 
            if val is None or val == "X" or val == -1:
                tensor[0, r, c, 0] = 1 
            elif isinstance(val, int):
                channel_idx = min(max(val + 1, 1), 9)
                tensor[0, r, c, channel_idx] = 1
    return tensor

def get_deterministic_moves(visible_board, size):
    safe_moves = set()
    known_hazards = set()
    clue_equations = [] 
    
    for r in range(size):
        for c in range(size):
            clue = visible_board[r, c]
            if clue <= 0: continue 
            
            unknowns = set()
            hazards_found = 0
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        if visible_board[nr, nc] == -1:
                            unknowns.add((nr, nc))
                        elif visible_board[nr, nc] == -2:
                            hazards_found += 1
            
            if not unknowns: continue
                
            remaining_hazards = clue - hazards_found
            clue_equations.append((remaining_hazards, unknowns))
            
            if remaining_hazards == len(unknowns):
                known_hazards.update(unknowns)
            if remaining_hazards == 0:
                safe_moves.update(unknowns)

    for req1, set1 in clue_equations:
        for req2, set2 in clue_equations:
            if set1 and set2 and set1 != set2 and set1.issubset(set2):
                difference_set = set2 - set1
                difference_req = req2 - req1
                
                if difference_req == len(difference_set):
                    known_hazards.update(difference_set)
                elif difference_req == 0:
                    safe_moves.update(difference_set)
                    
    return list(safe_moves), list(known_hazards)

def generate_field(size, hazards):
    field = np.zeros((size, size), dtype=int)
    indices = np.random.choice(size*size, hazards, replace=False)
    field.ravel()[indices] = -1
    
    clues = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            if field[r, c] == -1:
                clues[r, c] = -1
                continue
            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and field[nr, nc] == -1:
                        count += 1
            clues[r, c] = count
    return field, clues

def simulate_single_game(size, hazards):
    field, clues = generate_field(size, hazards)
    visible_board = np.full((size, size), -1) 
    safe_tiles_to_clear = (size * size) - hazards
    cleared_count = 0
    total_flagged = 0 
    
    while True:
        r, c = np.random.randint(0, size), np.random.randint(0, size)
        if field[r, c] != -1:
            visible_board[r, c] = clues[r, c]
            cleared_count += 1
            break

    while cleared_count < safe_tiles_to_clear:
        if total_flagged == hazards:
            for r in range(size):
                for c in range(size):
                    if visible_board[r, c] == -1:
                        visible_board[r, c] = clues[r, c]
            
            telemetry = {
                "cleared_percent": 100.0,
                "engine": "Constraint Solver",
                "explanation": "All hazards were successfully isolated and flagged. Remaining tiles were mathematically verified as safe."
            }
            return True, visible_board.flatten().tolist(), telemetry
            
        safe_moves, known_hazards = get_deterministic_moves(visible_board, size)
        
        for hr, hc in known_hazards:
            if visible_board[hr, hc] != -2:
                visible_board[hr, hc] = -2 
                total_flagged += 1 
            
        if safe_moves:
            r, c = safe_moves[0]
            move_type = "Deterministic Logic"
            reasoning = f"Calculated absolute safety at coordinate ({r}, {c}) using subset overlapping clues."
        else:
            if model is None: 
                return False, visible_board.flatten().tolist(), {}
                
            flat_board = visible_board.flatten().tolist()
            encoded = encode_board(flat_board)
            
            try:
                predictions = model.predict(encoded, verbose=0)[0].flatten()
            except Exception:
                predictions = np.random.rand(size * size)
            
            for i in range(size * size):
                row, col = divmod(i, size)
                if visible_board[row, col] != -1: 
                    predictions[i] = 999.0 
                    
            best_move_idx = np.argmin(predictions)
            r, c = divmod(best_move_idx, size)
            
            risk_pct = predictions[best_move_idx] * 100
            move_type = "CNN Intuition (Probabilistic)"
            reasoning = f"Logic failed. The CNN evaluated the board and selected ({r}, {c}) as the safest option with a predicted risk of {risk_pct:.1f}%. It was a forced guess."
        
        if field[r, c] == -1:
            flat_board = visible_board.flatten().tolist()
            flat_board[r * size + c] = "X"
            percent = (cleared_count / safe_tiles_to_clear) * 100
            
            telemetry = {
                "cleared_percent": round(percent, 1),
                "engine": move_type,
                "explanation": reasoning
            }
            return False, flat_board, telemetry
            
        if visible_board[r, c] == -1: 
            visible_board[r, c] = clues[r, c]
            cleared_count += 1
            
    telemetry = {"cleared_percent": 100.0, "engine": "Logic", "explanation": "Grid cleared."}
    return True, visible_board.flatten().tolist(), telemetry


# --- API MODELS ---

class BatchRequest(BaseModel):
    size: int
    hazards: int
    num_games: int

class ExplanationRequest(BaseModel):
    cleared_percent: float
    engine: str
    raw_explanation: str

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "MineSearcher AI Engine is running"}

@app.get("/initialize-mission")
def initialize(size: int = 10, hazards: int = 15):
    global current_field
    field, _ = generate_field(size, hazards)
    
    for r in range(size):
        for c in range(size):
            if field[r, c] == -1:
                continue
            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and field[nr, nc] == -1:
                        count += 1
            field[r, c] = count
            
    current_field = field
    return {"status": "Field Generated", "size": size}

@app.get("/scan-tile")
def scan_tile(r: int, c: int):
    global current_field
    if current_field is None:
        return {"error": "No mission initialized"}
    
    value = int(current_field[r, c])
    is_hazard = (value == -1)
    
    response = {
        "coordinate": [r, c],
        "sensor_reading": value,
        "is_hazard": is_hazard
    }
    
    # NEW: If a hazard is detonated, send back the entire board layout
    if is_hazard:
        response["full_field"] = current_field.flatten().tolist()
        
    return response

@app.post("/analyze-risk")
async def analyze_risk(data: dict):
    global model
    grid = data.get("grid")
    size = int(math.sqrt(len(grid)))
    
    if model is None:
        return {"heatmap": [0.1] * (size*size), "cnn_confidence": 0, "bayesian_est": 15, "safest_tile_index": -1}

    encoded = encode_board(grid)
    try:
        predictions = model.predict(encoded, verbose=0)[0] 
        heatmap = predictions.flatten().tolist()
    except Exception:
        heatmap = [0.1] * (size*size)
        
    # NEW: Find the optimal next move for the green UI highlight
    min_risk = float('inf')
    safest_idx = -1
    for i, val in enumerate(grid):
        if val is None: # Only consider unclicked tiles
            if heatmap[i] < min_risk:
                min_risk = heatmap[i]
                safest_idx = i
    
    avg_risk = float(np.mean(heatmap))
    
    return {
        "heatmap": heatmap,
        "cnn_confidence": round((1 - avg_risk) * 100, 2),
        "bayesian_est": round(avg_risk * 100, 2),
        "safest_tile_index": safest_idx
    }

@app.post("/batch-solve")
async def batch_solve(req: BatchRequest):
    wins = []
    losses = []
    
    for _ in range(req.num_games):
        won, final_board, telemetry = simulate_single_game(req.size, req.hazards)
        
        sim_data = {
            "board": final_board,
            "telemetry": telemetry
        }
        
        if won:
            wins.append(sim_data)
        else:
            losses.append(sim_data)
            
    win_rate = (len(wins) / req.num_games) * 100
    
    return {
        "win_rate": round(win_rate, 2),
        "total_games": req.num_games,
        "wins": wins,
        "losses": losses
    }

@app.post("/generate-explanation")
async def generate_explanation(req: ExplanationRequest):
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        return {"rich_explanation": "[API KEY MISSING] The system requires a valid Gemini API key to generate this report. Please add it to main.py."}

    try:
        prompt = f"""
        You are the tactical AI commander of an autonomous hazard-mapping rover.
        Analyze this post-mortem data from a recent grid scan:
        - Clearance Progress: {req.cleared_percent}%
        - Final Decision Engine Used: {req.engine}
        - Raw System Telemetry: {req.raw_explanation}

        Write a brief, 2-3 sentence dramatic but highly technical explanation of what happened. 
        If it reached 100%, praise the mathematical logic. If it failed, explain the uncertainty that forced the error. Do not use asterisks or markdown formatting in your response.
        """
        
        model = genai.GenerativeModel('gemma-2-27b-it')
        response = model.generate_content(prompt)
        
        return {"rich_explanation": response.text.strip()}
    except Exception as e:
        return {"rich_explanation": f"Failed to connect to Gemini API: {str(e)}"}