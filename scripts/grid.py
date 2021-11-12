#!/usr/bin/env python
# coding: utf-8

# SI units for all quantities!

import numpy as np
import matplotlib.pyplot as plt
from neighbor import check_neighbor
import sys 

dim_x = int(sys.argv[1])
dim_y = int(sys.argv[1])

x = np.linspace(-0.1, 0.1, dim_x)
y = np.linspace(-0.1, 0.1, dim_y)
dx = x[0] - x[1]

X,Y = np.meshgrid(x,y)
Z   = np.sqrt(X**2+Y**2)


# 4 circles with centers at (0.01+/-r1,0.01+/-r1) (within the square!) with a radius of R1
R1 = 0.075
r1 = 0.01
C1 = np.where((((X-(0.1-r1))**2+(Y-(0.1-r1))**2)  > (R1**2)) & 
             (((X-(0.1-r1))**2+(Y-(-0.1+r1))**2) > (R1**2)) &
             (((X-(-0.1+r1))**2+(Y-(-0.1+r1))**2)> (R1**2)) &
             (((X-(-0.1+r1))**2+(Y-(0.1-r1))**2) > (R1**2)) & 
             (Y<0.075) & (Y>-0.075) & (X<0.075) & (X>-0.075)
            )

# 4 circles with centers at along the axes at r2 (within the square!) with a radius of R2
R2 = 0.015
r2 = 0.025
C2 = np.where(((X**2+(Y-(0.1-r2))**2) < (R2**2))  & (Y>(0.075)))
C3 = np.where(((X**2+(Y-(-0.1+r2))**2) < (R2**2)) & (Y<(-0.075)))
C4 = np.where(((X-(0.1-r2))**2+Y**2 < (R2**2))    & (X>(0.075)))
C5 = np.where(((X-(-0.1+r2))**2+Y**2 < (R2**2))   & (X<(-0.075)))

# Combine all selected points 
S0 = np.concatenate((C1[0],C1[0],C2[0],C3[0],C4[0],C5[0]))
S1 = np.concatenate((C1[1],C1[1],C2[1],C3[1],C4[1],C5[1]))

# Sorted x and y indices
_x = []
_y = []

for i in range(len(S1)):
    _x.append(X[S0[i]][S1[i]])
    _y.append(Y[S0[i]][S1[i]])
    
_x = np.array(_x)    
_y = np.array(_y)    

# Sort indices, row major ordering
a = np.argsort(_y)
_x = _x[a]
_y = _y[a]
for i in np.unique(_y):
    xi = np.where(_y==i)
    xs = np.argsort(_x[xi])
    _x[xi] = _x[xi][xs]


# All points within the selected area
x = []
y = []

for i in range(len(_x)):
    xi = _x[i]
    yi = _y[i]
    if len(np.where(((xi-x)**2+(yi-y)**2)==0)[0])==0:
        x.append(xi)
        y.append(yi)
        
x = np.array(x)
y = np.array(y)


# Indices of the boundary elements
j = []
for i in range(len(x)):
    if check_neighbor(i,x,y)==0:
        j.append(i)

# Points on the boundary
bx = x[j]
by = y[j]
bx = np.array(bx)
by = np.array(by)
np.save('b-idx.npy', j)

# Indices of the boundary elements
j = []
for i in range(len(x)):
    if check_neighbor(i,x,y)!=0:
        j.append(i)

# Points in the interior
ix = x[j]
iy = y[j]
ix = np.array(ix)
iy = np.array(iy)

np.save('x-idx.npy', x)
np.save('y-idx.npy', y)

np.save('bx-idx.npy', bx)
np.save('by-idx.npy', by)

np.save('ix-idx.npy', ix)
np.save('iy-idx.npy', iy)

