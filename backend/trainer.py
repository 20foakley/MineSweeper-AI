import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

MINE = -9
HIDDEN = -1

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
    size = board.shape[0]
    tensor = np.zeros((size, size, 11), dtype=np.float32)

    for r in range(size):
        for c in range(size):
            val = board[r, c]

            if val == HIDDEN:
                tensor[r, c, 0] = 1
            else:
                tensor[r, c, val + 1] = 1

    tensor[:, :, 10] = compute_frontier(board)
    return tensor

def generate_field(size, hazards):
    minefield = np.zeros((size, size), dtype=np.int32)

    mine_indices = np.random.choice(size * size, hazards, replace=False)
    minefield.ravel()[mine_indices] = MINE

    clues = np.zeros((size, size), dtype=np.int32)

    for r in range(size):
        for c in range(size):
            if minefield[r, c] == MINE:
                clues[r, c] = MINE
                continue

            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        if minefield[nr, nc] == MINE:
                            count += 1

            clues[r, c] = count

    return minefield, clues


def reveal(clues, visible, r, c):
    """
    Recursive zero expansion.
    """
    size = clues.shape[0]

    if not (0 <= r < size and 0 <= c < size):
        return

    if visible[r, c] != HIDDEN:
        return

    if clues[r, c] == MINE:
        return  # never reveal mines during training

    visible[r, c] = clues[r, c]

    # expand zeros recursively
    if clues[r, c] == 0:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                reveal(clues, visible, r + dr, c + dc)

def simulate_game(clues, minefield, num_moves=None):
    """
    Simulate a realistic game in progress.
    """
    size = clues.shape[0]
    visible = np.full((size, size), HIDDEN, dtype=np.int32)

    safe_cells = list(zip(*np.where(minefield != MINE)))

    # first click guaranteed safe
    first_click = safe_cells[np.random.randint(len(safe_cells))]
    reveal(clues, visible, first_click[0], first_click[1])

    if num_moves is None:
        num_moves = np.random.randint(1, 20)

    for _ in range(num_moves):
        hidden_safe = [
            (r, c)
            for r, c in safe_cells
            if visible[r, c] == HIDDEN
        ]

        if not hidden_safe:
            break

        move = hidden_safe[np.random.randint(len(hidden_safe))]
        reveal(clues, visible, move[0], move[1])

    return visible



def get_training_batch(count=1000, size=10, hazards=15):
    X, Y = [], []

    for _ in range(count):
        minefield, clues = generate_field(size, hazards)

        visible_board = simulate_game(clues, minefield)

        X.append(encode_board(visible_board))
        Y.append((minefield == MINE).astype(np.float32)[..., np.newaxis])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    return X, Y

def build_hazard_cnn(size=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(size, size, 11)),

        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    return model

class MinesweeperGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=32, size=10, hazards=15, steps=1000):
        self.batch_size = batch_size
        self.size = size
        self.hazards = hazards
        self.steps = steps

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        return get_training_batch(
            count=self.batch_size,
            size=self.size,
            hazards=self.hazards
        )

# --- EXECUTION CODE ---
if __name__ == "__main__":
    import os

    SIZE = 10
    HAZARDS = 15
    MODEL_PATH = "hazard_model.keras"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    monitor="loss",
    mode="min"
    )

    train_gen = MinesweeperGenerator(
        batch_size=512,
        size=SIZE,
        hazards=HAZARDS,
        steps=100
    )

    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("Creating new model...")
        model = build_hazard_cnn(SIZE)

    model.fit(
        train_gen,
        epochs=10,
        callbacks=[checkpoint]

    )

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")