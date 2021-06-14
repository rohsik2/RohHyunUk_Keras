import numpy as np

org_data = np.array(range(1,11))

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    x = aaa[:, :-1]
    y = aaa[:, -1]
    return x, y
x, y = split_x(org_data, size)
print(x,y)