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

size=50
betaJ=0.2
n=1000

@njit(cache=True)
def main():
	M=[]
	L=2*random.randint(0,2,(size,size))-ones(size)
	U=energy(L)
	for k in range(n):
		i=random.randint(size)
		j=random.randint(size)
		dE=4*inter(L,i,j)
		if exp(-betaJ*dE)>random.uniform(0,1):
			L[i,j]=-L[i,j]
			U+=dE
		M.append(L.copy())
	return M
a=time.time()
M=main()

def main2():
	plt.ion()
	figure,ax=plt.subplots(figsize=(10,8))
	line1=ax.imshow(M[0])
	for i,L in enumerate(M):
		line1.set_data(L)
		figure.canvas.draw()
		figure.canvas.flush_events()
		print(i)

main2()
b=time.time()
print(b-a)