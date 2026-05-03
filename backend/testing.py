import numpy as np
from tensorflow.keras import layers, models


model = models.Sequential([
        layers.Input(shape=(10, 10, 10)),
        layers.Conv2D(75, (4,4), padding='same', activation='relu', strides=1),
        layers.Conv2D(75, (4,4), padding='same', activation='relu', strides=1),
        layers.Conv2D(75, (4,4), padding='same', activation='relu', strides=1),
        layers.Conv2D(1, (1,1), activation='softsign', padding='same',strides=1)
    ])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.summary()

# generate 9x9 board 
size = 9
board = np.zeros((size, size),dtype=np.int32)
print(board)
# place 10 mines randomly
rng = np.random.default_rng() # new numpy random generator
mine_indices = rng.choice(size*size, size = 10, replace = False)
board.ravel()[mine_indices] = -1
print(board)

# generate clue numbers for each cell

for r in range(size):
    for c in range(size):
        print(r,c)
        if board[r][c] == -1:
            continue
    # for each cell, see how many mines are adjacent to it
        count_neighbors = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    # no need to check current cell
                    continue
                row_adj = r+dr
                col_adj = c+dc
                
                print(f"yoyo {row_adj} {col_adj}")
                if 0 <= row_adj < size and 0 <= col_adj < size and board[row_adj][col_adj] == -1:
                    # inc for each adjacent mine
                    count_neighbors+=1
        board[r][c]=count_neighbors
print(board)

################################################

# encode_board
# each cell should get a 1x10 tensor
tensor = np.zeros((size,size,10))
print(tensor)
# indexing goes like tensor[x][y][z]
tensor[0][0][0]
for r in range(size):
    for c in range(size):
        value = board[r][c] #-1 thru 8
        tensor[r][c][value+1] = 1 # encode z dimension (zth stack)
print(tensor)