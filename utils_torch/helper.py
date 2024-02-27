# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:52:20 2022

@author: yc2634
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# helper functions
def odd_even_vector(nx):
    if (nx % 2) == 0 :
        nx_2 = nx / 2.
        k = np.linspace(1-nx_2, nx_2, nx)
    else:
        nx_2 = (nx-1) / 2.
        k = np.linspace(-nx_2, nx_2, nx)

    return k



# helper function
def plot_vfield(vfield, m0, J, inc, isl=0, vmin=None, vmax=None):
    
    mpl.rc('text', usetex=True)

    vmagnitude = np.sqrt( vfield[:,:,:,0]**2 + vfield[:,:,:,1]**2 + vfield[:,:,:,2]**2 )
    if vmin is None:
        vmin = np.min(vmagnitude.flatten())
    if vmax is None:
        vmax = np.max(vmagnitude.flatten())
    
    id_dir = np.argmax( np.abs(J) )
    plt.figure()
    # plt.imshow(vfield[:,:,isl,id_dir],cmap='viridis')
    # plt.imshow(vfield[:,:,isl,id_dir],cmap='rainbow')
    im = plt.imshow(vmagnitude, vmin=vmin, vmax=vmax ,cmap='rainbow')
    # cb = plt.colorbar(label=r'\textbf{a label}')
    cb = plt.colorbar()
    plt.axis('off')
    im.set_interpolation('none')
    cb.set_label('velocity magnitude',size=18,family='Times New Roman',rotation='vertical')
    cb.ax.tick_params(labelsize=18) 
    
    x, y, z = np.meshgrid(np.linspace(0, m0.L[0], m0.nx),
                          np.linspace(0, m0.L[1], m0.ny),
                          np.linspace(0, m0.L[2], m0.nz))
    x = np.moveaxis(x, 0, 1)
    y = np.moveaxis(y, 0, 1)
    z = np.moveaxis(z, 0, 1)
    figy = x[::inc,::inc,isl].flatten() / m0.dx
    figx = y[::inc,::inc,isl].flatten() / m0.dy
    figvy = -vfield[::inc,::inc,isl,0].flatten() / m0.dx
    figvx = vfield[::inc,::inc,isl,1].flatten() / m0.dy
    plt.quiver(figx, figy, figvx, figvy, color='black')

from skimage.morphology import binary_erosion
from skimage.morphology import disk
def plot_vfield_zonewise(vfield, m0, x0,y0, radius, J, inc, isl=0, vmin=None, vmax=None):
    mpl.rc('text', usetex=True)

    vmagnitude = np.sqrt( vfield[:,:,:,0]**2 + vfield[:,:,:,1]**2 + vfield[:,:,:,2]**2 )
    if vmin is None:
        vmin = np.min(vmagnitude.flatten())
    if vmax is None:
        vmax = np.max(vmagnitude.flatten())
    
    id_dir = np.argmax( np.abs(J) )
    plt.figure()
    # plt.imshow(vfield[:,:,isl,id_dir],cmap='viridis')
    # plt.imshow(vfield[:,:,isl,id_dir],cmap='rainbow')
    im = plt.imshow(vmagnitude, vmin=vmin, vmax=vmax ,cmap='rainbow')
    # cb = plt.colorbar(label=r'\textbf{a label}')
    cb = plt.colorbar()
    plt.axis('off')
    im.set_interpolation('none')
    cb.set_label('velocity magnitude',size=18,family='Times New Roman',rotation='vertical')
    cb.ax.tick_params(labelsize=18) 
    
    x, y, z = np.meshgrid(np.linspace(0, m0.L[0], m0.nx),
                          np.linspace(0, m0.L[1], m0.ny),
                          np.linspace(0, m0.L[2], m0.nz))
    x = np.moveaxis(x, 0, 1)
    y = np.moveaxis(y, 0, 1)
    z = np.moveaxis(z, 0, 1)
    x = x[::inc,::inc,isl].flatten()
    y = y[::inc,::inc,isl].flatten() 
    figy = x / m0.dx
    figx = y / m0.dy
    figvy = -vfield[::inc,::inc,isl,0].flatten() / m0.dx
    figvx = vfield[::inc,::inc,isl,1].flatten() / m0.dy
    
    num_solids = len(m0.label_solid)
    id_tot = np.zeros(figx.shape, dtype=bool)
    for i in range(num_solids):
        # s = np.zeros((inc*2+1,inc*2+1,1))
        # s[:,:,isl] = disk(inc)
        # coordi = np.where( binary_erosion(m0.Ifn==m0.label_solid[i],selem=s) )
        # id = np.isin(figx.astype(int), coordi[0]) & np.isin(figy.astype(int), coordi[1])
                
        id = (x-x0[0])**2+(y-y0[0])**2 < radius[i]**2*0.9

        plt.quiver(figx[id], figy[id], figvx[id], figvy[id], color='white')
        
        id_tot = id_tot | id

    plt.quiver(figx[~id_tot], figy[~id_tot], figvx[~id_tot], figvy[~id_tot], color='black')


