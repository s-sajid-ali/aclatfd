#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy.sparse as spsparse
import matplotlib.pyplot as plt
from neighbor import check_neighbor


# Load all, inner and boundary points

x = np.load('x-idx.npy')
y = np.load('y-idx.npy')

ix = np.load('ix-idx.npy')
iy = np.load('iy-idx.npy')

bx = np.load('bx-idx.npy')
by = np.load('by-idx.npy')


dx = x[1] - x[0]


# Lists to hold the matrix in CSR format
elements = []
rowidx   = []
colidx   = []

for i in range(len(x)):
    xi = x[i]
    yi = y[i]

    # check if point is in the interior region
    if len(np.intersect1d(np.where(xi==ix)[0], np.where(yi==iy)[0])):
        neighbors = check_neighbor(i,x,y)
        for j in neighbors[0]:
            if i!=j:
                elements.append(1/(dx**2))
                rowidx.append(i)
                colidx.append(j)
            if i==j:
                elements.append(-4/(dx**2))
                rowidx.append(i)
                colidx.append(i)
    # if point is on the boundary
    else:
        elements.append(1)
        rowidx.append(i)
        colidx.append(i)
        
elements = np.array(elements)
rowidx   = np.array(rowidx)
colidx   = np.array(colidx)


A = spsparse.csr_matrix((elements, (rowidx, colidx)), shape=(x.shape[0],x.shape[0]), dtype=np.float64)
spsparse.save_npz('A.npz', A)
