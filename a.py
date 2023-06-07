from numpy import *
from numba import njit

@njit
def a(p):
	return cumsum(p)

print(a(0.5))