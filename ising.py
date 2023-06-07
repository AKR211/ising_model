from numpy import *
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json

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
	return -U

#metropolis algorithm
@njit(cache=True)
def main(size,betaJ,n):
	Magnetization=[]
	Energy=[]                 
	L=2*random.randint(0,2,(size,size))-ones(size)
	L_init=L.copy()
	History=[(size,size)]*n
	U=energy(L)
	S=L.sum()
	for k in range(n):
		i=random.randint(size)
		j=random.randint(size)
		dE=4*interaction(L,i,j)  #change in energy
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
	betaJ=0.7                  #beta*J
	n=1000000                 #number of MC steps

	#plotting
	Energy,Lattice,Magnetization,L_init,History=main(size,betaJ,n)
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

	#saving
	name=str(input('Save?:'))
	if name!='':
		file=open(name+'.json','w')
		file.write(json.dumps([size,L_init.tolist()]+History))