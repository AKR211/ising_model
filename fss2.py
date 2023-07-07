from numpy import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

l=[4,5,6,7,8,9,10,11,12]
t=[3.02, 2.8, 2.7, 2.62, 2.58, 2.56, 2.5, 2.48, 2.42]
y=1/array(t)
x=1/array(l)

plt.plot(x,y,'.',color='r')
f=lambda L,betamax,c,k: betamax + c*(L**-k)
betamax,c,k=curve_fit(f,1/x,y,maxfev=2000)[0]
g=lambda L: betamax + c*(L**-k)
plt.plot(x,g(1/x))
plt.title(f'betamax={betamax.round(5)},c={c.round(5)},k={k.round(5)}')
plt.show()