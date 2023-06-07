from numpy import *
from numba import njit,jit
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json

@njit(cache=True)
def interaction(L,i,j):
	n=L.shape[0]
	return (L[i-1,j]+L[i,j-1]+L[(i+1)%n,j]+L[i,(j+1)%n])*L[i,j]

@njit(cache=True)
def energy(L):
	U=0
	for index,value in ndenumerate(L):
			U+=interaction(L,index[0],index[1])
	return -U

name=str(input('Filename:'))
file=open(name+'.json','r')
History=json.load(file)

size=History[0]
n=len(History)-2

@jit(nopython=False, cache=True)
def main():
	Magnetization=[]
	Energy=[]
	L=array(History[1]).reshape(size,size)
	L_init=L.copy()
	U=energy(L)
	S=L.sum()
	for a in History[2:]:
		i=a[0]
		j=a[1]
		if i!=size:
			U+=4*interaction(L,i,j)
			L[i,j]=-L[i,j]
			S+=2*L[i,j]
		Magnetization.append(S)
		Energy.append(U)
	return Energy,L,Magnetization,L_init

Energy,Lattice,Magnetization,L_init=main()
fig=plt.figure(figsize=([10,6]))
gs = gridspec.GridSpec(2, 2)
gs.update(wspace = 0.3, hspace = 0.3)
ax1=plt.subplot(gs[:2,0])
ax2=plt.subplot(gs[0,1])
ax3=plt.subplot(gs[1,1])
ax1.imshow(Lattice)
ax2.plot(range(1,n+1),Energy)
ax3.plot(range(1,n+1),Magnetization)
plt.show()