import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def encode_board(board):
    size = board.shape[0]
    tensor = np.zeros((size, size, 10))
    for r in range(size):
        for c in range(size):
            val = board[r, c]
            if val == -1: # Unknown
                tensor[r, c, 0] = 1
            elif 0 <= val <= 8: # Revealed Number
                tensor[r, c, int(val) + 1] = 1
    return tensor

def get_training_batch(count=1000, size=10, hazards=15):
    X, Y = [], []
    for _ in range(count):
        # 1. Create Ground Truth
        field = np.zeros((size, size))
        mine_indices = np.random.choice(size*size, hazards, replace=False)
        field.ravel()[mine_indices] = -1
        
        # Calculate actual clues for the ground truth
        full_clues = np.zeros((size, size))
        for r in range(size):
            for c in range(size):
                if field[r,c] == -1: 
                    full_clues[r,c] = -1
                    continue
                count_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size and field[nr, nc] == -1:
                            count_neighbors += 1
                full_clues[r,c] = count_neighbors

        # 2. Create Visible View (The AI's input)
        # Randomly reveal 20-40% of the board to simulate a game in progress
        visible_mask = np.random.rand(size, size) < np.random.uniform(0.2, 0.4)
        visible_board = np.full((size, size), -1)
        visible_board[visible_mask] = full_clues[visible_mask]
        
        X.append(encode_board(visible_board))
        Y.append((field == -1).astype(int)) 
    
    return np.array(X), np.array(Y)

def build_hazard_cnn(size=10):
    model = models.Sequential([
        layers.Input(shape=(size, size, 10)),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.Conv2D(1, (1,1), activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- EXECUTION CODE ---
if __name__ == "__main__":
    SIZE = 10
    HAZARDS = 15
    SAMPLES = 20000 # Start with 20k for faster training on a laptop

    print(f"Generating {SAMPLES} training boards...")
    X_train, Y_train = get_training_batch(SAMPLES, SIZE, HAZARDS)

    print("Building and training CNN...")
    model = build_hazard_cnn(SIZE)
    
    # Train the model
    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Save the model file
    model.save('hazard_model.h5')
    print("SUCCESS: hazard_model.h5 has been saved to your SSD.")