import numpy as np
from scipy import interpolate
import numpy.ma as ma
from scipy.interpolate import splprep, splev

def addTwo(a,b):
  return(a+b)
  
def splprep_predict(x,y,t):

  x = np.asarray(x)
  y = np.asarray(y)
  t = np.asarray(t)
  
  x = ma.masked_where(abs(x - 999) < 1e-4, x)
  y = ma.masked_where(abs(y - 999) < 1e-4, y)
  
  if(np.any(x.mask)):
    x = x[~x.mask]
  
  if(np.any(y.mask)):
    y = y[~y.mask]
  
  newX = [x[0]]
  newY = [y[0]]
  
  for i in range(min(len(x),len(y)))[1:]:
    if (abs(x[i] - x[i-1]) > 1e-4) or (abs(y[i] - y[i-1]) > 1e-4):
      newX.append(x[i])
      newY.append(y[i])	
      
  newX = np.asarray(newX)
  newY = np.asarray(newY)
  
  tck, u = splprep([newX, newY], s=0)
  newU = np.arange(0,1,t[1]/(t[-1]+t[1]))
  new_points = splev(newU, tck)
  
  return(new_points)

print("Finished Running Python")
