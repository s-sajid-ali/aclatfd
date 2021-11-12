#!/usr/bin/env python
# coding: utf-8

# Reference: `{SYNERGIA2_DIR, devel-3}/src/analysis_tools/beam_ploy.py`

# Arguments: input filename, output filename

import numpy as np
import matplotlib.pyplot as plt
import h5py, sys
import scipy.sparse as spsparse
import scipy.sparse.linalg as sparsela
from scipy.interpolate import griddata

# Get filename
infile  = sys.argv[1]
outfile = sys.argv[2]

# Coords format used by synergia
coords = {}
coords['x']      = 0
coords['xp']     = 1
coords['y']      = 2
coords['yp']     = 3
coords['cdt']    = 4
coords['dpop']   = 5
coords['pz']     = 7
coords['energy'] = 8
coords['t']      = 9
coords['z']      = 10


# Extract particles from hdf5 file
c = 299792458.0
hcoord = 'x'
vcoord = 'y'
h5        = h5py.File(infile, 'r')
particles = h5.get('particles')
npart     = particles.shape[0]
mass      = h5.get('mass')[()]
p_ref     = h5.get('pz')[()]
pz        = (p_ref * (1.0 + particles[:,5])).reshape(npart, 1)
betagamma = p_ref/mass
gamma     = np.sqrt(betagamma**2 + 1)
beta      = betagamma/gamma
energy    = np.sqrt(pz*pz + mass**2).reshape(npart, 1)
time      = (particles[:,4]*1.0e9/c).reshape(npart,1)
z         = (particles[:,4]*beta).reshape(npart,1)
particles = np.hstack((particles, pz, energy,time, z))

# Get 2D charge density on synergia grid
x = particles[:,coords[hcoord]]
y = particles[:,coords[vcoord]]
H, xedges, yedges = np.histogram2d(x, y, bins=50)
_xedges = (xedges[0:-1] + xedges[1:])/2.0
_yedges = (yedges[0:-1] + yedges[1:])/2.0
X, Y = np.meshgrid(xedges, yedges)

# Load lattice element finite difference
# discretization grid
x = np.load('x-idx.npy')
y = np.load('y-idx.npy')
ix = np.load('ix-idx.npy')
iy = np.load('iy-idx.npy')
bx = np.load('bx-idx.npy')
by = np.load('by-idx.npy')
bidx = np.load('b-idx.npy')

# Transfer charge density onto new grid,
# zero the charger density on boundaries
_xe, _ye = np.meshgrid(_xedges, _yedges)
charge_density = griddata((_xe.flatten(), _ye.flatten()), H.flatten(), (x, y), method='nearest')
charge_density[bidx] = 0

# Save charge density array
np.save(outfile, charge_density)
