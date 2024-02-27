import numpy as np
import matplotlib.pyplot as plt
from utils_torch.classes import microstructure, load_fluid_condition, param_algo
from utils_torch.brinkman_mod import *
from utils_torch.math_fcts import *

import time
import tracemalloc



#============ 2D unit cell, ellipsoid inclusion, reguler packing ==============
vxsiz = 1e-3
L1, L2, L3 = 1, 6.6, 1*vxsiz
nx, ny, nz = int(L1/vxsiz), int(L2/vxsiz), 1

x0 = [L1/2.]
y0 = [L2/2.]
b  = [2.2]
a  = [0.25]

Ifn = np.zeros([nx, ny, nz], dtype=np.uint8) + 0
x, y, z = np.meshgrid(np.linspace(0, L1, nx),
                      np.linspace(0, L2, ny),
                      np.linspace(0, L3, nz))
x = np.moveaxis(x, 0, 1)
y = np.moveaxis(y, 0, 1)
z = np.moveaxis(z, 0, 1)
zID = list()
id = np.sqrt( (x-x0[0])**2/a[0]**2 + (y-y0[0])**2/b[0]**2 ) <= 1.
Ifn[id] = 1
zID.append( 1 )
    
v_tow = np.pi * a[0]*b[0]/L1/L2
print('tow fraction: ', str(v_tow))


# micro permeability tensor
Rf = 0.023
vp = 0.53
Kp = 16/(9*np.pi*np.sqrt(6)) * (np.sqrt(0.91/(1-vp))-1)**2.5 * Rf**2
ks = np.array([[Kp, Kp, Kp, 0, 0, 0]])

# fibre orientation vectors
ex = np.array([[0,0,1]]) #fibre orientation
ey = np.array([[0,1,0]]) #transverse direction


# fluid viscosity
mu = 1.
mue = 1.

# reference parameters
phi0  = ( mu + mue ) / 2.
beta0 = ( 0 + mu/ks[0][0] ) / 2.
k0 = 0.
#=============================================================================

# #============ 2D unit cell, ellipsoid inclusion, hexa packing ==============
# vxsiz = 1e-2
# L1, L2, L3 = 1.65, 8, 1*vxsiz
# nx, ny, nz = int(L1/vxsiz), int(L2/vxsiz), 1

# x0 = [L1/2., 0, 0, L1, L1]
# y0 = [L2/2., 0, L2, 0, L2]
# b  = [2.2, 2.2, 2.2, 2.2, 2.2]
# a  = [0.25, 0.25, 0.25, 0.25, 0.25]

# Ifn = np.zeros([nx, ny, nz], dtype=np.uint8) + 0
# x, y, z = np.meshgrid(np.linspace(0, L1, nx),
#                       np.linspace(0, L2, ny),
#                       np.linspace(0, L3, nz))
# x = np.moveaxis(x, 0, 1)
# y = np.moveaxis(y, 0, 1)
# z = np.moveaxis(z, 0, 1)
# zID = list()
# for i in range(5):
#     id = np.sqrt( (x-x0[i])**2/a[i]**2 + (y-y0[i])**2/b[i]**2 ) <= 1.
#     Ifn[id] = i+1
#     zID.append( i+1 )
    
# v_tow = np.pi * a[0]*b[0]/L1/L2 *2
# print('tow fraction: ', str(v_tow))


# # micro permeability tensor
# Rf = 0.023
# vp = 0.53
# Kp = 16/(9*np.pi*np.sqrt(6)) * (np.sqrt(0.91/(1-vp))-1)**2.5 * Rf**2
# ks = np.array([[Kp, Kp, Kp, 0, 0, 0],
#                 [Kp, Kp, Kp, 0, 0, 0],
#                 [Kp, Kp, Kp, 0, 0, 0],
#                 [Kp, Kp, Kp, 0, 0, 0],
#                 [Kp, Kp, Kp, 0, 0, 0]])

# # fibre orientation vectors
# ex = np.array([[0,0,1],
#                 [0,0,1],
#                 [0,0,1],
#                 [0,0,1],
#                 [0,0,1]]) #fibre orientation
# ey = np.array([[0,1,0],
#                 [0,1,0],
#                 [0,1,0],
#                 [0,1,0],
#                 [0,1,0]]) #transverse direction


# # fluid viscosity
# mu = 1.
# mue = 1.

# # reference parameters
# phi0  = ( mu + mue ) / 2.
# beta0 = ( 0 + mu/ks[0][0] ) / 2.
# k0 = 0.
# #=============================================================================



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
p0 = param_algo(  cv_criterion = 1e-6,           #convergence criterion
                reference_phi0 = phi0,
                reference_beta0 = beta0,
                          itMax = 100000,           #max number of iterations
                        cv_acc = True,
                      AA_depth = 10
                )



# ========== load direction =================
lst_J = [[-1,0,0],
          [0,-1,0],
          [0,0,-1]];
# lst_J = [ [0,-1,0] ];
# ===========================================


# ------------------------- 3 independent simulations ------------------------
Htensor = np.zeros((3,3))
Ktensor = np.zeros((3,3))

for J in lst_J:

    print('******************  J =',str(J),' ******************')
    id_dir = np.argmax(np.array(J)!=0)

    # ------------------------ load & fluid condition -----------------------
    l0 = load_fluid_condition( macro_load = J,     #gradient of pressure
                                      viscosity = mu,    #fluid viscosity
                                viscosity_solid = mue,   #fluid viscosity in solid region
                              )
    
    # ------------------------------- solution -------------------------------

    tracemalloc.start()
    
    # H, vfield, gmacro = brinkman_fft_solver_stress(m0, l0, p0, freqType='modified', freqLaplacian='classical')
    H, vfield, gmacro = brinkman_fft_solver_velocity(m0, l0, p0, freqType='modified', freqLaplacian='classical')
    K, vfield, vmacro = brinkman_fft_solver_velocityP(m0, l0, p0, freqType='modified', freqLaplacian='classical')
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB\n")
    tracemalloc.stop()


    # -------------------------- record into tensor --------------------------
    #
    Htensor[:,id_dir] = H
    #
    # Ktensor[:,id_dir] = K
    

Ktensor = np.linalg.inv(Htensor)
print('Normalised permeability K/(ab): \n\n', str(Ktensor/a[0]/b[0]))
