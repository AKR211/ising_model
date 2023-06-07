from numpy import *
from numba import njit,jit
import matplotlib.pyplot as plt
import time

@njit(cache=True)
def inter(L,i,j):
	n=L.shape[0]
	return (L[(i-1)%n,j]+L[i,(j-1)%n]+L[(i+1)%n,j]+L[i,(j+1)%n])*L[i,j]

@njit(cache=True)
def energy(L):
	U=0
	for i in range(L.shape[0]):
		for j in range(L.shape[0]):
				U+=inter(L,i,j)
	return -U

size=100
betaJ=10
n=1000000

@jit(nopython=False, cache=True)
def main():
	L=2*random.randint(0,2,(size,size))-ones(size)
	U=energy(L)
	plt.ion()
	figure, ax = plt.subplots(figsize=(10, 8))
	line1 = ax.imshow(L)
	for k in range(n):
		i=random.randint(size)
		j=random.randint(size)
		dE=4*inter(L,i,j)
		if exp(-betaJ*dE)>random.uniform(0,1):
			L[i,j]=-L[i,j]
			U+=dE
		line1.set_data(L)
		figure.suptitle(f'counter={k}',x=0.5,y=0.95,fontsize=36)
		figure.canvas.draw()
		figure.canvas.flush_events()
main()