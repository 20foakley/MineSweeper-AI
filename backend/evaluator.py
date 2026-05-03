'''

import numpy as np
import tensorflow as tf
import time

print("Loading Hazard AI Model...")
try:
    model = tf.keras.models.load_model('hazard_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def encode_board(board, size=10):
    """
    Converts the current grid into a 10-channel tensor.
    Channel 0: Unknown tiles.
    Channels 1-9: Revealed numbers (0-8 clues).
    """
    tensor = np.zeros((1, size, size, 10))
    for r in range(size):
        for c in range(size):
            val = board[r, c]
            if val == -1: # Unknown
                tensor[0, r, c, 0] = 1
            elif 0 <= val <= 8: # Revealed Number
                channel_idx = min(max(int(val) + 1, 1), 9)
                tensor[0, r, c, channel_idx] = 1
    return tensor

def generate_field(size=10, hazards=15):
    """
    Generates a ground-truth hazard field and the corresponding clue map.
    """
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

def get_deterministic_moves(visible_board, size=10):
    """
    Applies strict spatial logic AND Advanced Subset Logic (Overlaps).
    """
    safe_moves = set()
    known_hazards = set()
    
    # Store clues as: (remaining_hazards_needed, set_of_unknown_coords)
    clue_equations = [] 
    
    # 1. Standard Logic & Equation Gathering
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
            
            # Rule 1: All unknowns are hazards
            if remaining_hazards == len(unknowns):
                known_hazards.update(unknowns)
            
            # Rule 2: All unknowns are safe
            if remaining_hazards == 0:
                safe_moves.update(unknowns)

    # 2. Advanced Subset Logic (The 1-2 Pattern Solver)
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

def play_autonomous_game(size=10, hazards=15):
    """
    Simulates a full game using Advanced Logic + CNN Fallback.
    """
    field, clues = generate_field(size, hazards)
    visible_board = np.full((size, size), -1) 
    
    safe_tiles_to_clear = (size * size) - hazards
    cleared_count = 0
    total_flagged = 0 
    
    # Force a safe first move
    while True:
        r, c = np.random.randint(0, size), np.random.randint(0, size)
        if field[r, c] != -1:
            visible_board[r, c] = clues[r, c]
            cleared_count += 1
            break

    while cleared_count < safe_tiles_to_clear:
        # GLOBAL WIN CONDITION: All hazards found
        if total_flagged == hazards:
            for r in range(size):
                for c in range(size):
                    if visible_board[r, c] == -1:
                        visible_board[r, c] = clues[r, c]
                        cleared_count += 1
            break 
            
        # --- PHASE 1: Advanced Deterministic Logic ---
        safe_moves, known_hazards = get_deterministic_moves(visible_board, size)
        
        # Mark known hazards
        for hr, hc in known_hazards:
            if visible_board[hr, hc] != -2:
                visible_board[hr, hc] = -2 
                total_flagged += 1 
            
        if safe_moves:
            # Take the first guaranteed safe move
            r, c = safe_moves[0]
        else:
            # --- PHASE 2: CNN Intuition Fallback ---
            encoded = encode_board(visible_board, size)
            predictions = model.predict(encoded, verbose=0)[0].flatten()
            
            for i in range(size * size):
                row, col = divmod(i, size)
                if visible_board[row, col] != -1: 
                    predictions[i] = 1.0 
                    
            best_move_idx = np.argmin(predictions)
            r, c = divmod(best_move_idx, size)
        
        # --- Evaluate the Move ---
        if field[r, c] == -1:
            return False # Detonated
            
        if visible_board[r, c] == -1: 
            visible_board[r, c] = clues[r, c]
            cleared_count += 1
        
    return True 

# --- Execution Block ---
if __name__ == "__main__":
    GAMES_TO_PLAY = 10
    wins = 0
    
    print(f"\nStarting Autonomous Evaluation: {GAMES_TO_PLAY} Missions...")
    print("Agent Architecture: Neuro-Symbolic V2 (Subset Logic + CNN Fallback)")
    print("-" * 50)
    
    start_time = time.time()
    
    for i in range(1, GAMES_TO_PLAY + 1):
        if play_autonomous_game():
            wins += 1
        
        if i % 10 == 0:
            print(f"Mission {i}/{GAMES_TO_PLAY} | Current Win Rate: {(wins/i)*100:.1f}%")
            
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Total Missions:         {GAMES_TO_PLAY}")
    print(f"Successful Clearances:  {wins}")
    print(f"Final Agent Win Rate:   {(wins/GAMES_TO_PLAY)*100:.2f}%")
    print(f"Time Elapsed:           {total_time:.2f} seconds")
    print("=" * 50)
    '''