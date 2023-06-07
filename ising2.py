from numpy import *
from numba import njit
import matplotlib.pyplot as plt

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
betaJ=0.7
n=1000000
L=2*random.randint(0,2,(size,size))-ones(size)

@njit(cache=True)
def main1(L):
	M1,M2=[],[]
	U=energy(L)
	for k in range(n):
		U_=energy(L)
		i=random.randint(size)
		j=random.randint(size)
		dE=4*inter(L,i,j)
		L_=copy(L)
		L_[i,j]=-L[i,j]
		p=exp(-betaJ*dE)
		r=random.uniform(0,1)
		if p>r:
			L=L_
			U+=dE
		M1.append(U)
		M2.append(U_)
	return M1,M2

@njit(cache=True)
def main2(L):
	M=[]
	
	for k in range(n):
		L_=copy(L)
		i=random.randint(size)
		j=random.randint(size)
		L_[i,j]=-L[i,j]
		p=exp(-betaJ*4*inter(L,i,j))
		r=random.uniform(0,1)
		if p>r:
			L=L_
		M.append(U)
	return M

M1,M2=main1(L)
plt.plot(range(1,n+1),M1)
plt.plot(range(1,n+1),M2)
plt.show()