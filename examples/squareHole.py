import numpy as np
import matplotlib.pyplot as plt
from utils_torch.classes import microstructure, load_fluid_condition, param_algo
from utils_torch.brinkman_mod import *
from utils_torch.math_fcts import *

import time
import tracemalloc


#========================= 2D unit cell, with open hole =======================
L1, L2 = 1, 1
nx, ny, nz = 50, 50, 1
L3 = 1/nx

radius = [L1 * 0.4]
x0 = [L1/2.]
y0 = [L2/2.]

Ifn = np.zeros([nx, ny, nz], dtype=np.uint8) + 1
zID = [1]
x, y, z = np.meshgrid(np.linspace(0, L1, nx),
                      np.linspace(0, L2, ny),
                      np.linspace(0, L3, nz))
x = np.moveaxis(x, 0, 1)
y = np.moveaxis(y, 0, 1)
z = np.moveaxis(z, 0, 1)
for i in range(len(x0)):
    id = np.sqrt( (x-x0[i])**2 + (y-y0[i])**2 ) <= radius[i]
    Ifn[id] = 0


# micro permeability tensor
ks = np.array([[1e-8, 1e-8, 1e-8, 0, 0, 0]])

ang0 = np.pi/4
ex = np.array([[np.cos(ang0),np.sin(ang0),0]])
ey = np.array([[-np.sin(ang0),np.cos(ang0),0]])


# fluid viscosity
mu = 1
mue = mu

# reference parameters
phi0  = ( mu + mue ) / 2.
beta0 = ( 0 + mu/np.min(ks[:,:3]) ) / 2.
#=============================================================================


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
                          itMax = 1000000,           #max number of iterations
                        cv_acc = True,
                      AA_depth = 4,
                      AA_increment = 4
                )

# -------------------------- loading parameters ----------------------------
l0 = load_fluid_condition(      macro_load = [1,0,0],     #macro velocity
                                  viscosity = mu,    #fluid viscosity
                            viscosity_solid = mue,   #fluid viscosity in solid region
                          )

############################
#### macro permeability ####
############################
# p0.cv_acc = False
# tracemalloc.start()
# H1, vfield1, gmacro1 = brinkman_fft_solver_stress(m0, l0, p0, freqType='modified', freqLaplacian='classical')
# K1 = 1./H1[0]
# current, peak = tracemalloc.get_traced_memory()
# print(f"\nAlgo1: Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
# tracemalloc.stop()

p0.cv_acc = True
p0.AA_depth = 8
p0.AA_increment = 4
tracemalloc.start()
H2, vfield2, gmacro2 = brinkman_fft_solver_velocity(m0, l0, p0, freqType='modified', freqLaplacian='classical')
K2 = 1./H2[0]
current, peak = tracemalloc.get_traced_memory()
print(f"\nAlgo2: Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
tracemalloc.stop()


p0.cv_acc = True
p0.AA_depth = 8
p0.AA_increment = 8
l0.macro_load = [-1,0,0]
tracemalloc.start()
K3, vfield3, vmacro3 = brinkman_fft_solver_velocityP(m0, l0, p0, freqType='modified', freqLaplacian='classical')
K3 = K3[0]
current, peak = tracemalloc.get_traced_memory()
print(f"\nAlgo3: Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
tracemalloc.stop()    
    
# print('K1, K2, K3: %.12e, %.12e, %.12e'%(K1, K2, K3))
print('K2, K3: %.12e, %.12e'%(K2, K3))



# #
# fig, ax = plt.subplots()
# ax.plot(vfield1[int(nx/2),:,0,0])
# ax.plot(vfield2[int(nx/2),:,0,0])

# ax.plot(vfield3[int(nx/2),:,0,0])


# fig, ax = plt.subplots()
# ax.plot(vfield1[:,int(nx/2),0,0])
# ax.plot(vfield2[:,int(nx/2),0,0])
# ax.plot(vfield3[int(nx/2),:,0,0])


