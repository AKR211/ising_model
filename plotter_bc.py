from ising import data1,data2,average,linspace,array,sign,gridspec,var,mean,std,argwhere,diff,sign
import matplotlib.pyplot as plt
from numba import njit,jit
import json

@njit
def values(size,n,p,eq,Temps):
	Es=[0.0]
	Ms=[0.0]
	BC=[0.0]
	for Temp in Temps:
		data=data1(size,1/Temp,n,p)
		M=array(data[1])
		M4=average(M**4)
		M2=average(M**2)
		BC.append(1-(M4/(3*(M2**2))))
		Es.append(average(data[0][int(eq*n):]))
		Ms.append(-abs(average(data[1][int(eq*n):])))
	Ms,Es,BC=array(Ms)/(size**2),array(Es)/(size**2),array(BC)
	return Es[1:],Ms[1:],BC[1:]

def plot1():
	Temps=linspace(2,2.5,101)[1:]
	size=500
	n=1000000
	p=1
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
	ax1.set(xlabel='Temperature', ylabel='|M|/N')
	ax2=plt.subplot(gs[1,0])
	ax2.set_title('Binder Cumulant')
	ax2.set(xlabel='Temperature', ylabel='Binder Cumulant (U)')
	ax3=plt.subplot(gs[1,1])
	ax3.set_title('Relative BC')
	ax3.set(xlabel='Temperature', ylabel='Ui/Uj')
	
	var=[4,8,12,16]
	BCs=[]
	for x in var:
		size=x
		colour=colours.pop()
		Es,Ms,BC=values(size,n,p,eq,Temps)
		ax0.plot(Temps,Es,'.',color=colour) #values("parameter"=x)
		ax1.plot(Temps,Ms,'.',color=colour)
		ax2.plot(Temps,BC,'.',color=colour)
		BCs.append(BC.copy())
		if len(var)>1:
			ax0.legend(var)
			ax1.legend(var)
			ax2.legend(var)
	f1=(BCs[3]/BCs[1])
	f2=(BCs[0]/BCs[2])
	f3=(BCs[0]/BCs[0])
	ax3.plot(Temps,f1,'-',color='red')
	ax3.plot(Temps,f2,'-',color='blue')
	ax3.plot(Temps,f3,'-',color='black')
	ax3.legend(['U16/U8','U4/U12','1'])
	print(Temps[argwhere(diff(sign(f1 - f3))).flatten()])

plot1()
plt.show()