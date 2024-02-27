import numpy as np
import matplotlib.pyplot as plt
from utils_torch.classes import microstructure, load_fluid_condition, param_algo
from utils_torch.brinkman_mod import *
from utils_torch.math_fcts import *

import time
import tracemalloc

# helper
def parallelChannel(ks, mu, mue, G, ny, a, b):
    
    ud = -G * ks / mu
    lc = np.sqrt(mue*ks/mu)
    
    y = np.linspace(-b, a, ny)
    yf = y[y>=0]
    yp = y[y<=0]
        
    uf = ud - G*yf*(a-yf)/2/mu - G*a*lc/2/mue/np.tanh(b/2/lc)
    
    # up = ud - G*a*lc/2/mue * (np.exp((b+yp)/lc)+np.exp(-yp/lc)) / (np.exp(b/lc)-1)
    up = ud - G*a*lc/2/mue * (np.exp(yp/lc)+np.exp(-(yp+b)/lc)) / (1-np.exp(-b/lc))

    return np.concatenate((up,uf)), np.concatenate((yp,yf))
# ======


# ======================================
a = 0.1
b = 0.1
ny = 320
dy = (a+b)/ny
Lx, Ly, Lz = dy, a+b, dy

y = np.linspace(-b, a, ny)
Ifn = np.zeros([1, ny, 1], dtype=np.uint8)
Ifn[:,y<=0,:] = 1

# micro permeability tensor
ks = np.array([[1e-6, 1e-6, 1e-6, 0, 0, 0]])

ks_min = list()
for i in range(1):
    ks_min.append( np.linalg.eig([[ks[i][0],ks[i][3],ks[i][4]],
                                  [ks[i][3],ks[i][1],ks[i][5]],
                                  [ks[i][4],ks[i][5],ks[i][2]]])[0].min() )
ks_min = np.min(ks_min)


ang0 = 0
ex = np.array([[np.cos(ang0),np.sin(ang0),0]])
ey = np.array([[-np.sin(ang0),np.cos(ang0),0]])

# fluid viscosity
mu = 1.
mue = 0.2

# reference parameters
phi0  = ( mu + mue ) / 2.
beta0 = ( 0 + mu/ks_min ) / 2.



# ----------------------------- microstructure -------------------------------
m0 = microstructure(          Ifn = Ifn,         #labeled image
                                L = [Lx,Ly,Lz], #physical dimension of RVE
                      label_fluid = 0,          #label for fluid region
                      label_solid = [1],         #label for solid(porous) region 
               micro_permeability = ks,         #local permeability
                       local_axes = [ex, ey]
                    )

# -------------------------- algorithm parameters ----------------------------
p0 = param_algo(   cv_criterion = 1e-6,           #convergence criterion
                 reference_phi0 = phi0,
                reference_beta0 = beta0,
                          itMax = 10000,           #max number of iterations
                         cv_acc = True,
                       AA_depth = 10
                )

# ------------------------ load & fluid condition -----------------------
l0 = load_fluid_condition(      macro_load = [-1,0,0],     #macro load
                                 viscosity = mu,           #fluid viscosity
                           viscosity_solid = mue,          #fluid viscosity in solid region
                          )

# ------------------------- full-field simulations ------------------------

# tracemalloc.start()
# l0.macro_load = [1,0,0]
# H1, vfield1, gmacro1 = brinkman_fft_solver_stress(m0, l0, p0, freqType='classical', freqLaplacian='classical')
# current, peak = tracemalloc.get_traced_memory()
# print(f"Algo1: Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
# tracemalloc.stop()

tracemalloc.start()
l0.macro_load = [1,0,0]
p0.cv_acc = True
p0.AA_depth = 10
H2, vfield2, gmacro2 = brinkman_fft_solver_velocity(m0, l0, p0, freqType='modified', freqLaplacian='classical')
current, peak = tracemalloc.get_traced_memory()
print(f"\nAlgo2: Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
tracemalloc.stop()

tracemalloc.start()
l0.macro_load = [-1,0,0]
p0.cv_acc = True
p0.AA_depth = 10
K3, vfield3, vmacro3 = brinkman_fft_solver_velocityP(m0, l0, p0, freqType='modified', freqLaplacian='classical')
current, peak = tracemalloc.get_traced_memory()
print(f"\nAlgo3: Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
tracemalloc.stop()


# ==============================
# Compare to analytical solution
# ==============================
import matplotlib

font = {'family'  : 'Times New Roman',
        'weight'  : 'normal',
        'size'    : 36}
matplotlib.rc('font', **font)


# analytical solution
G = -1
v0, y0 = parallelChannel(ks[0][0], mu, mue, G, 10000, a, b)
fig, ax = plt.subplots()
# fig.set_figwidth(8)
# fig.set_figheight(6)
# fig.set_linewidth(20)
plt.rcParams["figure.figsize"] = (6,7)
plt.tight_layout()
ax.set_xscale("log")
ax.plot(v0, y0,'-k', linewidth=2)

# # numerical solution - algo1
# sc = gmacro1[0] / G # scale
# vfield1 = vfield1 / sc
# ax.plot(vfield1[0,:,0,0], y, '--')

# numerical solution - algo2
sc = gmacro2[0] / G # scale
vfield2 = vfield2 / sc
ax.plot(vfield2[0,:,0,0], y, '-.', linewidth=2)

# numerical solution - algo3
ax.plot(vfield3[0,:,0,0], y, '--', linewidth=2)


# ax.set_xlim(5e-7,5e-1)
ax.yaxis.set_ticks((-0.1,0,0.1))
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.xaxis.set_ticks((1e-6,1e-5,1e-4,1e-3)) #ks=1e-6
# ax.xaxis.set_ticks((1e-4,1e-3)) #ks=1e-4
# ax.xaxis.set_ticks((1e-8,1e-6,1e-4)) #ks=1e-8

plt.show()
