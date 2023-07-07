from numpy import array,random
import json

def lattice_builder(size,p=0.5): #p = probability of '1'
	return random.choice([1,-1],(size,size),p=[p,1-p])


size=50
p=0.5

if __name__=='__main__':
	lattice=lattice_builder(size,p)
	file=open('lattice.json','w')
	file.write(json.dumps(lattice.tolist()))
	
	print(lattice.shape)

