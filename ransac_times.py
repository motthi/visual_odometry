import numpy as np

ita = 0.8
eps = 0.1
s = 4

K = np.log(1 - ita) / np.log(1 - (1 - eps)**s)


print(f"ita: {ita}")
print(f"eps: {eps}")
print(f"s: {s}")
print(f"K: {K}")
