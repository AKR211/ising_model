from numpy import *
from numba import njit,jit
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json

@njit(cache=True)
def lattice_builder(size,p=0.5): #p = probability of '1'
	return 2*searchsorted(cumsum(array([p,1-p])), random.rand(size,size))-1

#interaction energy of one lattice site
@njit(cache=True)
def interaction(L,i,j):
	n=L.shape[0]
	return (L[i-1,j]+L[i,j-1]+L[(i+1)%n,j]+L[i,(j+1)%n])*L[i,j]

#total energy of the lattice
@njit(cache=True)
def energy(L):
	U=0
	for index,value in ndenumerate(L):
			U+=interaction(L,index[0],index[1])
	return -U/2

#metropolis algorithm
@njit(cache=True)
def main(size,betaJ,n,p=0.5):
	Magnetization=[]
	Energy=[]                 
	L=lattice_builder(size,p)
	L_init=L.copy()
	History=[(size,size)]*n
	U=energy(L)
	S=L.sum()
	for k in range(n):
		i=random.randint(size)
		j=random.randint(size)
		dE=2*interaction(L,i,j)  #change in energy
		if exp(-betaJ*dE)>random.uniform(0,1):  #acceptance condition
			L[i,j]=-L[i,j]
			U+=dE
			S+=2*L[i,j]
			History[k]=(i,j)
		Magnetization.append(S)
		Energy.append(U)
	return Energy,L,Magnetization,L_init,History

#plotting and saving
if __name__=='__main__':

	#values of parameters
	size=500                   #lattice size
	Temp=3                   #Temperature
	n=10000000                 #number of MC steps
	p=0.5

	#plotting
	Energy,Lattice,Magnetization,Lattice_init,History=main(size,1/Temp,n,p)
	fig=plt.figure(figsize=([10,6]))
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace = 0.3, hspace = 0.5)
	ax0=plt.subplot(gs[0,0])
	ax0.set_title('Initial Lattice')
	ax1=plt.subplot(gs[1,0])
	ax1.set_title('Final Lattice')
	ax2=plt.subplot(gs[0,1])
	ax2.set_title('Energy')
	ax3=plt.subplot(gs[1,1])
	ax3.set_title('Magnetization')
	ax0.imshow(Lattice_init, cmap='Greys')
	ax1.imshow(Lattice, cmap='Greys')
	ax2.plot(range(1,n+1),Energy)
	ax2.set(xlabel='MC steps', ylabel='Energy')
	ax3.plot(range(1,n+1),Magnetization)
	ax3.set(xlabel='MC steps', ylabel='Magnetization')
	plt.show()

	#saving
	name=str(input('Save?:'))
	if name!='':
		file=open(name+'.json','w')
		file.write(json.dumps([size,L_init.tolist()]+History))