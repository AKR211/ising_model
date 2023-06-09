from ising import data1,data2,average,linspace,array,sign,gridspec,var
import matplotlib.pyplot as plt
from numba import njit,jit
import json

@njit
def ctl_diff(X,Y,i):
	return(Y[i+1]-Y[i-1])/(X[i+1]-X[i-1])

@jit(nopython=False)
def values(size,n,p,eq,Temps):
	Es=[]
	Ms=[]
	for Temp in Temps:
		data=data1(size,1/Temp,n,p)
		Es.append(average(data[0][int(eq*n):]))
		Ms.append(-abs(average(data[1][int(eq*n):])))
	Ms,Es=array(Ms)/(size**2),array(Es)/(size**2)
	return Es,Ms

def plot1():
	Temps=[i/20 for i in range(1,101)]
	size=50
	n=1000000
	p=0.5
	eq=0.75
	
	fig=plt.figure(figsize=([10,6]))
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace = 0.5, hspace = 0.5)
	colours=['black','violet','blue','green','orange','red']
	ax0=plt.subplot(gs[0,0])
	ax0.set_title('Energy')
	ax0.set(xlabel='Temperature', ylabel='E/N')
	ax1=plt.subplot(gs[0,1])
	ax1.set_title('Magnetization')
	ax1.set(xlabel='Temperature', ylabel='M/N')
	ax2=plt.subplot(gs[1,0])
	ax2.set_title('Heat Capacity')
	ax2.set(xlabel='Temperature', ylabel='Cv/N')
	ax3=plt.subplot(gs[1,1])
	ax3.set_title('Susceptibility')
	ax3.set(xlabel='Temperature', ylabel='Chi/N')
	
	var=[50]
	for x in var:
		size=x
		colour=colours.pop()
		Es,Ms=values(size,n,p,eq,Temps)
		ax0.plot(Temps,Es,'.',color=colour) #values("parameter"=x)
		ax1.plot(Temps,Ms,'.',color=colour)
		ax2.plot(Temps[1:-1],[ctl_diff(Temps,Es,i) for i in range(len(Temps[1:-1]))],'.',color=colour)
		ax3.plot(Temps[1:-1],[ctl_diff(Temps,Ms,i) for i in range(len(Temps[1:-1]))],'.',color=colour)
		if len(var)>1:
			ax0.legend(var)
			ax1.legend(var)
			ax2.legend(var)
			ax3.legend(var)
	return [size,n,p,eq,var,Es.tolist(),Ms.tolist()]

def plot2():
	size=500
	n=10000000
	p=0.5
	
	fig=plt.figure(figsize=([10,6]))
	var=[50,250,500]
	gs = gridspec.GridSpec(2, len(var))
	gs.update(wspace = 0.3, hspace = 0.3)
	for i,x in enumerate(var):
		size=x
		L=data2(size,0.7,n,p)
		ax=plt.subplot(gs[0,i])
		ax.axis('off')
		ax.set_title(f'size={x}')
		ax.imshow(L, cmap='Greys')
	for i,x in enumerate(var):
		size=x
		L=data2(size,0.2,n,p)
		ax=plt.subplot(gs[1,i])
		ax.axis('off')
		ax.set_title(f'size={x}')
		ax.imshow(L, cmap='Greys')
	return [size,n,p,var,L.tolist()]

save=plot1()
fig=plt.gcf()
plt.show()
name=str(input('Save?:'))
if name!='':
	file=open(name+'.json','w')
	file.write(json.dumps(save))
	fig.savefig(name+'.png')