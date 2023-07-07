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
def interaction(L,i,j,delta):
	n=L.shape[0]
	return (L[(i-1+delta[0])%n,(j+delta[1])%n]+L[(i+delta[0])%n,(j-1+delta[1])%n]+L[(i+1+delta[0])%n,(j+delta[1])%n]+L[(i+delta[0])%n,(j+1+delta[1])%n])*L[i,j]

#total energy of the lattice
@njit(cache=True)
def energy(L,delta):
	U=0
	for index,value in ndenumerate(L):
			U+=interaction(L,index[0],index[1],delta)
	return -U/2

#metropolis algorithm
@njit(cache=True)
def main(size,betaJ,n,p=0.5,delta=[0,0]):
	Magnetization=[]
	Energy=[]                 
	L=lattice_builder(size,p)
	L_init=L.copy()
	U=energy(L,delta)
	S=L.sum()
	for k in range(n):
		i=random.randint(size)
		j=random.randint(size)
		dE=interaction(L,i,j,array(delta))+interaction(L,i,j,-array(delta))  #change in energy
		if exp(-betaJ*dE)>random.uniform(0,1):  #acceptance condition
			L[i,j]=-L[i,j]
			U+=dE
			S+=2*L[i,j]
		Magnetization.append(S)
		Energy.append(U)
	return Energy,Magnetization,L_init,L

@njit(cache=True)
def data1(size,betaJ,n,p=0.5,delta=[0,0]):
	Magnetization=[]
	Energy=[]                 
	L=lattice_builder(size,p)
	U=energy(L,delta)
	S=L.sum()
	for k in range(n):
		i=random.randint(size)
		j=random.randint(size)
		dE=interaction(L,i,j,array(delta))+interaction(L,i,j,-array(delta))  #change in energy
		if exp(-betaJ*dE)>random.uniform(0,1):  #acceptance condition
			L[i,j]=-L[i,j]
			U+=dE
			S+=2*L[i,j]
		Magnetization.append(S)
		Energy.append(U)
	return Energy,Magnetization

@njit(cache=True)
def data2(size,betaJ,n,p=0.5):               
	L=lattice_builder(size,p)
	for k in range(n):
		i,j=random.randint(size),random.randint(size)
		if exp(-betaJ*2*interaction(L,i,j))>random.uniform(0,1):  #acceptance condition
			L[i,j]=-L[i,j]
	return L

#plotting and saving
if __name__=='__main__':

	#values of parameters
	size=500                   #lattice size
	Temp=1                   #Temperature
	n=10000000                 #number of MC steps
	p=0.5

	#plotting
	Energy,Magnetization,Lattice_init,Lattice=main(size,1/Temp,n,p,[1,0])
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