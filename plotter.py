from ising import data1,data2,average,linspace,array,sign,gridspec
import matplotlib.pyplot as plt
import json

def values(size,n,p,eq,Temps):
	Es=[]
	Ms=[]
	for Temp in Temps:
		data=data1(size,1/Temp,n,p)
		Es.append(average(data[0][int(eq*n):]))
		Ms.append(average(data[1][int(eq*n):]))
	return array(Es),array(Ms)

def plot1():
	Temps=[i/20 for i in range(1,101)]
	size=50
	n=1000000
	p=0.75
	eq=0.75
	
	fig=plt.figure(figsize=([10,6]))
	gs = gridspec.GridSpec(1, 2)
	gs.update(wspace = 0.5, hspace = 0.5)
	colours=['black','violet','blue','green','orange','red']
	ax0=plt.subplot(gs[0,0])
	ax0.set_title('Energy')
	ax0.set(xlabel='Temperature', ylabel='Energy')
	ax1=plt.subplot(gs[0,1])
	ax1.set_title('Magnetization')
	ax1.set(xlabel='Temperature', ylabel='Magnetization')
	
	var=[250]
	for x in var:
		size=x
		colour=colours.pop()
		Es,Ms=values(size,n,p,eq,Temps)
		ax0.plot(Temps,Es/(2*size**2),'.',color=colour) #values("parameter"=x)
		ax1.plot(Temps,Ms/(size**2),'.',color=colour)
	return [size,n,p,eq,var,Es.tolist(),Ms.tolist()]

def plot2():
	size=250
	n=1000000
	p=0.25
	
	fig=plt.figure(figsize=([10,6]))
	var=[10000,100000,1000000,10000000]
	gs = gridspec.GridSpec(2, len(var))
	gs.update(wspace = 0.3, hspace = 0.3)
	for i,x in enumerate(var):
		n=x
		L=data2(size,0.7,n,p)
		ax=plt.subplot(gs[0,i])
		ax.axis('off')
		ax.set_title(f'n={x}')
		ax.imshow(L, cmap='Greys')
	for i,x in enumerate(var):
		n=x
		L=data2(size,0.2,n,p)
		ax=plt.subplot(gs[1,i])
		ax.axis('off')
		ax.set_title(f'n={x}')
		ax.imshow(L, cmap='Greys')
	return [size,n,p,var,L.tolist()]

save=plot2()
fig=plt.gcf()
plt.show()
name=str(input('Save?:'))
if name!='':
	file=open(name+'.json','w')
	file.write(json.dumps(save))
	fig.savefig(name+'.png')