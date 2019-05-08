import numpy as np
import scipy.io

xtemp=np.linspace(0,2*np.pi, 100)
ytemp=np.cos(xtemp)

scipy.io.savemat('test.mat', dict(x=xtemp, y=ytemp))

