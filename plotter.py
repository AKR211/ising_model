from ising import main,average,linspace,array,sign
import matplotlib.pyplot as plt

'''
{param_space}=[]
{const1}=
{const2}=
{const3}=
{values}=[]

for {param} in {param_space}:
	Energy,Lattice,Magnetization,Lattice_init,History=main(size,betaJ,n,p)
	{values}.append({value})

plt.plot({param_space},{values})
'''


Temps=[i/20 for i in range(1,101)]
size=50
n=1000000
p=0.6

def values(k):
	values=[]
	for Temp in Temps:
		data=main(size,1/Temp,n,p)
		values.append(average(data[k][int(0.75*n):]))
	return array(values)

values=values(0)
max=values[0]
betaJs_=[0.01]
values_=[max]
for k in range(1,len(values)):
	if values[k]<max:
		max=values[k]
		values_.append(max)
		betaJs_.append((1+k)/100)

plt.plot(Temps,values,'r.')
plt.show()