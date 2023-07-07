from numba import jit
from ising_fixed import data1,linspace,array,var,average
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def ctl_diff(X,Y,i):
	return(Y[i+1]-Y[i-1])/(X[i+1]-X[i-1])

@jit
def max_(size,n,p,eq,Temps):
	max=0
	idx=0
	for Temp in Temps:
		val=var(array(data1(size,1/Temp,n,p)[1][int(eq*n):]))
		if val>max:
			max=val
			idx=Temp
	print(idx)
	return idx,val

@jit
def main():
	Temps=linspace(2,3,51)[1:]
	sizes=array([8,16,24,32])
	Tmaxs=array([max_(size,100000,0.75,0.75,Temps)[0] for size in sizes])
	return sizes,1/Tmaxs

f=lambda L,betamax,c: betamax + c*(L**-1)
sizes,betas=main()
plt.scatter(1/sizes,betas)
plt.show()
print(curve_fit(f,sizes,betas,maxfev=2000))