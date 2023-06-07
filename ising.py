from numpy import *
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import gridspec

@njit(cache=True)
def interaction(L,i,j):
	n=L.shape[0]
	return (L[i-1,j]+L[i,j-1]+L[(i+1)%n,j]+L[i,(j+1)%n])*L[i,j]

@njit(cache=True)
def energy(L):
	U=0
	for i in range(L.shape[0]):
		for j in range(L.shape[0]):
				U+=interaction(L,i,j)
	return -U

size=500
betaJ=0.2
n=10000000

@njit(cache=True)
def main():
	S=[]
	M=[]
	L=2*random.randint(0,2,(size,size))-ones(size)
	U=energy(L)
	T=L.sum()
	for k in range(n):
		i=random.randint(size)
		j=random.randint(size)
		dE=4*interaction(L,i,j)
		if exp(-betaJ*dE)>random.uniform(0,1):
			L[i,j]=-L[i,j]
			U+=dE
			T+=2*L[i,j]
		S.append(T)
		M.append(U)
	return M,L,S

M,L,S=main()
fig=plt.figure(figsize=([10,6]))
gs = gridspec.GridSpec(2, 2)
gs.update(wspace = 0.3, hspace = 0.3)
ax1=plt.subplot(gs[:2,0])
ax2=plt.subplot(gs[0,1])
ax3=plt.subplot(gs[1,1])
ax1.imshow(L)
ax2.plot(range(1,n+1),M)
ax3.plot(range(1,n+1),S)
plt.show()