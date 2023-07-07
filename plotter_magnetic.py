from ising_magnetic import data1,data2,average,linspace,array,sign,gridspec,var,mean,std
import matplotlib.pyplot as plt
from numba import njit,jit
import json

@njit
def values(size,n,p,eq,Temps,B):
	Es=[0.0]
	Ms=[0.0]
	varEs=[0.0]
	varMs=[0.0]
	for Temp in Temps:
		data=data1(size,1/Temp,n,p,B)
		Es.append(average(data[0][int(eq*n):]))
		varEs.append(var(array(data[0][int(eq*n):])))
		Ms.append(-abs(average(data[1][int(eq*n):])))
		varMs.append(var(array(data[1][int(eq*n):])))
	Ms,Es,varEs,varMs=array(Ms)/(size**2),array(Es)/((2+B)*size**2),array(varEs),array(varMs)
	return Es[1:],Ms[1:],varEs[1:],varMs[1:]

def plot1():
	Temps=linspace(0,20,101)[1:]
	size=50
	n=1000000
	p=0.75
	eq=0.75
	B=1
	
	fig=plt.figure(figsize=([10,6]))
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace = 0.5, hspace = 0.5)
	colours=['black','violet','blue','green','orange','red']
	ax0=plt.subplot(gs[0,0])
	ax0.set_title('Energy')
	ax0.set(xlabel='Temperature', ylabel='E/N')
	ax1=plt.subplot(gs[0,1])
	ax1.set_title('Magnetization')
	ax1.set(xlabel='Temperature', ylabel='|M|/N')
	ax2=plt.subplot(gs[1,0])
	ax2.set_title('Heat Capacity')
	ax2.set(xlabel='Temperature', ylabel='Cv')
	ax3=plt.subplot(gs[1,1])
	ax3.set_title('Susceptibility')
	ax3.set(xlabel='Temperature', ylabel='Chi')
	
	var=[0,0.5,1,2]
	for x in var:
		B=x
		colour=colours.pop()
		Es,Ms,varEs,varMs=values(size,n,p,eq,Temps,B)
		ax0.plot(Temps,Es,'.',color=colour) #values("parameter"=x)
		ax1.plot(Temps,Ms,'.',color=colour)
		ax2.plot(Temps,varEs/((Temps)**2),color=colour)
		ax3.plot(Temps,varMs/Temps,color=colour)
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