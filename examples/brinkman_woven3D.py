import numpy as np
import matplotlib.pyplot as plt
from utils.classes import microstructure, load_fluid_condition, param_algo
from utils.brinkman_mod import *
from utils.math_fcts import *

import time
import tracemalloc





# ====================== TexGen generated microstrucgure ======================
import meshio

# import vtu data
fnameVTU    = r"woven_nest3layers_50x50x50.vtu"
data        = meshio.read(fnameVTU)
field_data  = data.cell_data
coordinates = data.points
origin = [coordinates[:,0].min(),coordinates[:,1].min(),coordinates[:,2].min()]

h1 = np.average(np.diff(np.unique(coordinates[:,0])))
h2 = np.average(np.diff(np.unique(coordinates[:,1])))
h3 = np.average(np.diff(np.unique(coordinates[:,2])))

L1 = coordinates[:,0].max() - coordinates[:,0].min()
L2 = coordinates[:,1].max() - coordinates[:,1].min()
L3 = coordinates[:,2].max() - coordinates[:,2].min()

n1 = round(L1/h1)
n2 = round(L2/h2)
n3 = round(L3/h3)

# label image
yarnID = np.reshape(field_data["YarnIndex"], (n3,n2,n1))
yarnID = np.swapaxes(yarnID, 0, 2)

zID = np.arange(np.count_nonzero(yarnID>=0)) + 1
nzones = len(zID)

Ifn = np.zeros((n1,n2,n3), dtype=np.uint64())
Ifn[yarnID>=0] = zID


# micro permeability tensor in local coordinates
ks0 = np.array([1e-6, 1e-7, 1e-7, 0, 0, 0]) #mm^(2)

ks = np.ones((nzones,6)) * ks0
ks_min = np.min(ks0[:3])

#local coordinate axis
ex  = np.reshape(field_data["Orientation"], (n3,n2,n1,3))
ex  = np.swapaxes(ex, 0, 2)
ex = ex[yarnID>=0,:]

# generate perpendicular vectors
axmax = np.argmax(ex, axis=-1)
axmid = 3 - axmax - np.argmin(ex, axis=-1)
ey = np.zeros((nzones,3))
for i in range(nzones):
    tmp = -ex[i,axmid[i]] / ex[i,axmax[i]]
    ey[i,axmax[i]] = tmp
    ey[i,axmid[i]] = 1
    ey[i,:] = ey[i,:] / np.sqrt(1+tmp**2)



# fluid viscosity
mu = 1.
mue = 1.

# reference parameters
phi0  = ( mu + mue ) / 2.
beta0 = ( 0 + mu/ks_min ) / 2.
k0 = 0
# =============================================================================

# ----------------------------- microstructure -------------------------------
m0 = microstructure(         Ifn = Ifn,         #labeled image
                                L = [L1,L2,L3], #physical dimension of RVE
                      label_fluid = 0,          #label for fluid region
                      label_solid = zID,        #label for solid(porous) region 
              micro_permeability = ks,         #local permeability
                      local_axes = [ex, ey]
                    )
del Ifn, zID, ks, ex, ey

# -------------------------- algorithm parameters ----------------------------
p0 = param_algo(   cv_criterion = 1e-6,           #convergence criterion
                 reference_phi0 = phi0,
                reference_beta0 = beta0,
                          itMax = 100000,           #max number of iterations
                )


# ------------------------ load & fluid condition -----------------------
l0 = load_fluid_condition( macro_load = [1,0,0],     #gradient of pressure
                                  viscosity = mu,    #fluid viscosity
                            viscosity_solid = mue,   #fluid viscosity in solid region
                          )


# p0.cv_acc = False
# tracemalloc.start()
# H1, vfield1, gmacro1 = brinkman_fft_solver_stress(m0, l0, p0, freqType='classical', freqLaplacian='classical')
# K1 = 1./H1[0]
# current, peak = tracemalloc.get_traced_memory()
# print(f"Algo1: Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
# tracemalloc.stop()

p0.cv_acc = True
p0.AA_depth = 8
p0.AA_increment = 4
tracemalloc.start()
H2, vfield2, gmacro2 = brinkman_fft_solver_velocity(m0, l0, p0, freqType='classical', freqLaplacian='classical')
K2 = 1./H2[0]
current, peak = tracemalloc.get_traced_memory()
print(f"Algo2: Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
tracemalloc.stop()


p0.cv_acc = True
p0.AA_depth = 8
p0.AA_increment = 8
l0.macro_load = [-1,0,0]
tracemalloc.start()
K3, vfield3, vmacro3 = brinkman_fft_solver_velocityP(m0, l0, p0, freqType='classical', freqLaplacian='classical')
K3 = K3[0]
current, peak = tracemalloc.get_traced_memory()
print(f"Algo3: Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
tracemalloc.stop()    
    
# print('K1, K2, K3: %.12e, %.12e, %.12e'%(K1, K2, K3))
print('K2, K3: %.12e, %.12e'%(K2, K3))




