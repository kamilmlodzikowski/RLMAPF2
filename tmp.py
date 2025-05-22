import numpy as np
obstacle_symbol = 'X'
empty_symbol = ' '

width = 16
array = np.array([[obstacle_symbol for _ in range(width+1)] for _ in range(width+1)])

# cut a circle with size width

for i in range(width):
    for j in range(width):
        if (i - width/2)**2 + (j - width/2)**2 < (width/2)**2:
            array[i][j] = empty_symbol

# add x on the right side


# print(array.tolist())
for row in array:
    print("".join(row))