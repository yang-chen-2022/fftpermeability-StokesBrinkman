import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .classes import grid
from .math_fcts import *

#from scipy.fft import fftn, ifftn
import torch
from torch.fft import fftn, ifftn
import time

torch.set_default_dtype(torch.float64)


"""
FFT solver for Darcy-Brinkman equation
                 vector-form Brinkman
         with Anderson's acceleration
--------------------------------------

written by Yang Chen, University of Bath, 2022.04.06

Modification Log
----------------
    2022.04.22 Add the option of Anderson's acceleration
    2022.04.22 Proper consideration for beta-beta0
    2022.12.30 re-written by YC
    2024.02.01 
        1. Add the option of using beta(x) as input, instead of ks(x)
        2. Add the option of initialising velocity field
        3. Use torch 
"""
def brinkman_fft_solver_velocity( m0,  # microstructure
                                  l0,  # load, fluid conditions
                                  p0,  # algorithm parameters
                                  freqType = None, 
                                  freqLaplacian = None,
                                  vfield0 = None,
                                  device = None
                                  ):
    
    if device is None:
        device = 'cpu'
    
    t = time.time()
        
    # reference material
    phi0  = p0.reference_phi0
    beta0 = p0.reference_beta0
    
    # define distribution funcions
    vmacro = torch.as_tensor(l0.macro_load).to(device)#macro velocity
    mu     = l0.viscosity           #viscosity in fluid region
    mue    = l0.viscosity_solid     #effective viscosity in porous regions

    if m0.micro_beta is None:
        #calculate the coefficient beta from ks
        ks     = m0.micro_permeability  #tensor in local coordinates
        ks_inv = inv_matrix3x3sym_vec(ks)
        
        lst_beta = mu * np.array(ks_inv)        #in the local coord syst.
        lst_beta = rot_tensorSym_loc2glob(lst_beta, m0.local_axes[0],m0.local_axes[1]) #in global coord syst.

        beta = np.zeros((m0.nx,m0.ny,m0.nz, 6))
        beta[m0.Ifn!=m0.label_fluid,:] = lst_beta         #beta in solid region
    else:
        beta = m0.micro_beta #- beta0
        print('  Calculation with BETA as input')
    
    # conver to torch tensor
    beta = torch.from_numpy(beta).to(device)
    
    #the coefficient phi
    phi = torch.zeros((m0.nx,m0.ny,m0.nz)).to(device)
    idx = np.where(m0.Ifn==m0.label_fluid)
    phi[idx] = mu
    idx = np.where(m0.Ifn!=m0.label_fluid)
    phi[idx] = mue
    
    # grid and frequency vectors
    g0 = grid(nx=m0.nx, ny=m0.ny, nz=m0.nz, dx=m0.dx, dy=m0.dy, dz=m0.dz)
    if (freqType is None):
        freqType = 'classical'
    freq = g0.initFREQ(freqType)
    freq = torch.from_numpy(freq).to(device)
    
    # laplacian operator freq*freq
    if freqLaplacian == 'modified':
        freqSquare = -g0.initFREQ_laplacian()
    else:
        freqSquare =  freq[:,:,:,0]**2 + freq[:,:,:,1]**2 + freq[:,:,:,2]**2
    
    
    # Green operator
    QQ = torch.zeros([g0.nx, g0.ny, g0.nz, 6]).to(device)
    freqSquare[0,0,0] = 1.
    QQ[:,:,:,0] = (1. - freq[:,:,:,0]*freq[:,:,:,0]/freqSquare)
    QQ[:,:,:,1] = (1. - freq[:,:,:,1]*freq[:,:,:,1]/freqSquare)
    QQ[:,:,:,2] = (1. - freq[:,:,:,1]*freq[:,:,:,2]/freqSquare)
    QQ[:,:,:,3] = (   - freq[:,:,:,0]*freq[:,:,:,1]/freqSquare)
    QQ[:,:,:,4] = (   - freq[:,:,:,0]*freq[:,:,:,2]/freqSquare)
    QQ[:,:,:,5] = (   - freq[:,:,:,1]*freq[:,:,:,2]/freqSquare)
    freqSquare[0,0,0] = 0.
    QQ[0,0,0,:] = 0.
    
    # allocate variables
    gma     = torch.zeros((m0.nx,m0.ny,m0.nz, 3)).to(device)
    tauF    = torch.zeros((m0.nx,m0.ny,m0.nz, 3), dtype=torch.complex128).to(device)
    vfield  = torch.zeros((m0.nx, m0.ny, m0.nz, 3)).to(device)
    vfieldF = torch.zeros((m0.nx,m0.ny,m0.nz, 3), dtype=torch.complex128).to(device)
    
    #gmaF    = torch.zeros((m0.nx,m0.ny,m0.nz, 3), dtype=torch.complex128).to(device)
    #residu  = torch.zeros((m0.nx, m0.ny, m0.nz, 3)).to(device)
    #residuF = torch.zeros((m0.nx,m0.ny,m0.nz, 3), dtype=torch.complex128).to(device)

    # initialisation: velocity field, laplacian of velocity
    for j0 in range(3):
        vfield[:,:,:,j0] = vmacro[j0]
            
    # variables for Anderson acceleration
    if p0.cv_acc == True:
        act_R = torch.zeros((m0.nx, m0.ny, m0.nz, 3, p0.AA_depth)).to(device)
        act_U = torch.zeros((m0.nx, m0.ny, m0.nz, 3, p0.AA_depth)).to(device)
    else:
        act_R = None
        act_U = None
        
    if vfield0 is None:
        vfield0 = vfield.clone()
    else:
        vfield = torch.from_numpy(vfield0).to(device).clone()
    
    # iterative solution
    iters = 1
    iact = 0
    iact1 = 0
    while True:
        
        # store for Anderson's acceleration
        if p0.cv_acc==True:
            act_U[:,:,:,:,iact] = vfield
            
                
        # Polarisation vectors : store tau in gma
        gma[:,:,:,0] = (phi-phi0)*gma[:,:,:,0] - ((beta[:,:,:,0]-beta0)*vfield[:,:,:,0] + \
                                                   beta[:,:,:,3]*vfield[:,:,:,1] + \
                                                   beta[:,:,:,4]*vfield[:,:,:,2] )
        gma[:,:,:,1] = (phi-phi0)*gma[:,:,:,1] - ( beta[:,:,:,3]*vfield[:,:,:,0] + \
                                                  (beta[:,:,:,1]-beta0)*vfield[:,:,:,1] + \
                                                   beta[:,:,:,5]*vfield[:,:,:,2] )
        gma[:,:,:,2] = (phi-phi0)*gma[:,:,:,2] - ( beta[:,:,:,4]*vfield[:,:,:,0] + \
                                                   beta[:,:,:,5]*vfield[:,:,:,1] + \
                                                  (beta[:,:,:,2]-beta0)*vfield[:,:,:,2] )
        
        # FFT
        for j0 in range(3):
            tauF[:,:,:,j0] = fftn(gma[:,:,:,j0])
        
        
        # apply the green operator -> TODO: calculate GG on the fly
        vfieldF[:,:,:,0] = ( QQ[:,:,:,0]*tauF[:,:,:,0] + \
                             QQ[:,:,:,3]*tauF[:,:,:,1] + \
                             QQ[:,:,:,4]*tauF[:,:,:,2] ) / (phi0*freqSquare+beta0)
        vfieldF[:,:,:,1] = ( QQ[:,:,:,3]*tauF[:,:,:,0] + \
                             QQ[:,:,:,1]*tauF[:,:,:,1] + \
                             QQ[:,:,:,5]*tauF[:,:,:,2] ) / (phi0*freqSquare+beta0)
        vfieldF[:,:,:,2] = ( QQ[:,:,:,4]*tauF[:,:,:,0] + \
                             QQ[:,:,:,5]*tauF[:,:,:,1] + \
                             QQ[:,:,:,2]*tauF[:,:,:,2] ) / (phi0*freqSquare+beta0)
         
        # enforce the loading - macro velocity
        vfieldF[0,0,0,0] = vmacro[0] * g0.ntot
        vfieldF[0,0,0,1] = vmacro[1] * g0.ntot
        vfieldF[0,0,0,2] = vmacro[2] * g0.ntot
        
        
        # IFFT
        for j0 in range(3):
            vfield[:,:,:,j0] = ifftn(vfieldF[:,:,:,j0]).real
            #gma[:,:,:,j0]  = ifftn(-freqSquare* vfieldF[:,:,:,j0]).real
            
        # Anderson's acceleration
        if p0.cv_acc==True:
            act_R[:,:,:,:,iact] = act_U[:,:,:,:,iact]-vfield
            
            # if iters % p0.AA_depth ==0:
            if iters % p0.AA_increment ==0:
                vfield = AAcceleration(act_R, act_U)
                ##update gamam with new vfield
                #for j0 in range(3):
                #    gma[:,:,:,j0]  = ifftn(-freqSquare* fftn(vfield[:,:,:,j0])).real
            #else:
            #    #update gamma with vfieldF
            #    for j0 in range(3):
            #        gma[:,:,:,j0]  = ifftn(-freqSquare* vfieldF[:,:,:,j0]).real
                
            if iters % p0.AA_depth == 0:
                iact = 0
            else:
                iact += 1
        #else:
        #    #update gamma with vfieldF
        #    for j0 in range(3):
        #        gma[:,:,:,j0]  = ifftn(-freqSquare* vfieldF[:,:,:,j0]).real
                
        
        # convergence check
        if p0.cv_acc==True:
            # if iters % p0.AA_depth == 0:
            if iters % p0.AA_increment == 0:
                pass #Force to do an additional iteration
            else:
                test = torch.sqrt( torch.sum((act_R[:,:,:,0,iact-1]**2 +
                                              act_R[:,:,:,1,iact-1]**2 +
                                              act_R[:,:,:,2,iact-1]**2).flatten())/g0.ntot )
                test = test / torch.sqrt(vmacro[0]**2+vmacro[1]**2+vmacro[2]**2)
        else:
            test = torch.sqrt(torch.sum( ((vfield[:,:,:,0]-vfield0[:,:,:,0])**2 +
                                          (vfield[:,:,:,1]-vfield0[:,:,:,1])**2 +
                                          (vfield[:,:,:,2]-vfield0[:,:,:,2])**2).flatten())/g0.ntot )
            test = test / torch.sqrt(torch.sum(vmacro**2))
        
        
        # pde residual: 
        #   R = (phi0*gma_{k} - beta0*vfield_{k} - pfield_{k} + tau_{k-1}) / ||Gmacro||_2
        test_eq = np.nan
        if iters%1==0:
            # Polarisation vector  (at iteration k-1)
            tau = torch.zeros_like(gma)
            for j0 in range(3):
                tau[...,j0] = ifftn(tauF[...,j0])
            
            # Laplacien of velocity (at iteration k)
            for j0 in range(3):
                gma[:,:,:,j0]  = ifftn(-freqSquare* fftn(vfield[:,:,:,j0])).real
            
            # pressure gradient field (at iteration k)
            pfield = torch.zeros_like(vfield)
            pfield[:,:,:,0] = phi*gma[:,:,:,0] - (beta[:,:,:,0]*vfield[:,:,:,0] + \
                                                  beta[:,:,:,3]*vfield[:,:,:,1] + \
                                                  beta[:,:,:,4]*vfield[:,:,:,2])
            pfield[:,:,:,1] = phi*gma[:,:,:,1] - (beta[:,:,:,3]*vfield[:,:,:,0] + \
                                                  beta[:,:,:,1]*vfield[:,:,:,1] + \
                                                  beta[:,:,:,5]*vfield[:,:,:,2])
            pfield[:,:,:,2] = phi*gma[:,:,:,2] - (beta[:,:,:,4]*vfield[:,:,:,0] + \
                                                  beta[:,:,:,5]*vfield[:,:,:,1] + \
                                                  beta[:,:,:,2]*vfield[:,:,:,2])
            # macro pressure gradient
            W = torch.as_tensor((pfield[...,0].mean(), pfield[...,1].mean(), pfield[...,2].mean()))
                                   
            # equilibrium residual
            eq_residual = phi0*gma - beta0*vfield - pfield + tau
            test_eq = torch.sqrt( torch.sum((eq_residual[:,:,:,0]**2 + \
                                             eq_residual[:,:,:,1]**2 + \
                                             eq_residual[:,:,:,2]**2).flatten())/g0.ntot )
            test_eq = test_eq / torch.sqrt( torch.sum(W**2) )
        
            
        # stop ?
        if test<p0.cv_criterion:
            print('  iteration %d -- residual: U-based (%.3e), E-based (%.3e)'%(iters, test, test_eq))
            break
        
        # counter
        iters += 1
        
        #
        if (iters % 100)==0:
            print('  iteration %d -- residual: U-based (%.3e), E-based (%.3e)'%(iters, test, test_eq))

        # to avoid infinite loop
        if iters>p0.itMax:
            print('Warning: number of iterations exceeds limit (%d)'%p0.itMax)
            print('  iteration %d -- residual: U-based (%.3e), E-based (%.3e)'%(iters, test, test_eq))
            break

        # update
        if p0.cv_acc==False:
            vfield0 = vfield.clone()
            
    
    # macro pressure gradient
    W = torch.zeros(3).to(device)
    W[0] = torch.sum( (beta[:,:,:,0]*vfield[:,:,:,0]+ \
                       beta[:,:,:,3]*vfield[:,:,:,1]+ \
                       beta[:,:,:,4]*vfield[:,:,:,2]).flatten() )/g0.ntot
    W[1] = torch.sum( (beta[:,:,:,3]*vfield[:,:,:,0]+ \
                       beta[:,:,:,1]*vfield[:,:,:,1]+ \
                       beta[:,:,:,5]*vfield[:,:,:,2]).flatten() )/g0.ntot
    W[2] = torch.sum( (beta[:,:,:,4]*vfield[:,:,:,0]+ \
                       beta[:,:,:,5]*vfield[:,:,:,1]+ \
                       beta[:,:,:,2]*vfield[:,:,:,2]).flatten() )/g0.ntot
    W[0] = -torch.sum( (phi*gma[:,:,:,0]).flatten() )/g0.ntot + W[0]
    W[1] = -torch.sum( (phi*gma[:,:,:,1]).flatten() )/g0.ntot + W[1]
    W[2] = -torch.sum( (phi*gma[:,:,:,2]).flatten() )/g0.ntot + W[2]
    
    
    # macro resistivity
    id_dir = np.argmax( np.abs(vmacro.cpu()) )
    H = W / mu / vmacro[id_dir]
    print('macro resistivity H: ', str(H))    
    
    #
    print('residuals: U-based (%.3e),  E-based (%.3e)'%(test, np.nan))
    
    #
    print('Total time (%f s);  Total nb of iters (%d)'%(time.time()-t, iters))

   
    
    return H.cpu().numpy(), vfield.cpu().numpy(), -W.cpu().numpy()






"""
FFT solver for Darcy-Brinkman equation
                 vector-form Brinkman
         with Anderson's acceleration
                     Pressure control
--------------------------------------

written by Yang Chen, University of Bath, 2022.04.06

Modification Log
----------------
    2022.04.22 Add the option of Anderson's acceleration
    2022.04.22 Proper consideration for beta-beta0
    2022.12.30 re-written by YC
    2024.02.01 
        1. Add the option of using beta(x) as input, instead of ks(x)
        2. Add the option of initialising velocity field
        3. Use torch 
"""
def brinkman_fft_solver_velocityP( m0,  # microstructure
                                   l0,  # load, fluid conditions
                                   p0,  # algorithm parameters
                                   freqType = None, 
                                   freqLaplacian = None,
                                   vfield0 = None,
                                   device = None
                                   ):
    if device is None:
        device = 'cpu'
        
    t = time.time()
        
    # reference material
    phi0  = p0.reference_phi0
    beta0 = p0.reference_beta0
    
    # define distribution funcions
    gmacro = torch.as_tensor(l0.macro_load).to(device)#macro pressure gradient
    mu     = l0.viscosity           #viscosity in fluid region
    mue    = l0.viscosity_solid     #effective viscosity in porous regions
    
    #the coefficient phi
    phi = np.zeros((m0.nx,m0.ny,m0.nz))
    phi[m0.Ifn==m0.label_fluid] = mu
    phi[m0.Ifn!=m0.label_fluid] = mue
    phi = torch.as_tensor(phi).to(device)
    
    if m0.micro_beta is None:
        #calculate the coefficient beta from ks
        ks     = m0.micro_permeability  #tensor in local coordinates
        ks_inv = inv_matrix3x3sym_vec(ks)
        
        lst_beta = mu * np.array(ks_inv)        #in the local coord syst.
        lst_beta = rot_tensorSym_loc2glob(lst_beta, m0.local_axes[0],m0.local_axes[1]) #in global coord syst.

        beta = np.zeros((m0.nx,m0.ny,m0.nz, 6))
        beta[m0.Ifn!=m0.label_fluid,:] = lst_beta         #beta in solid region
    else:
        beta = m0.micro_beta
    
    #
    beta = torch.as_tensor(beta).to(device)
    
    # grid and frequency vectors
    g0 = grid(nx=m0.nx, ny=m0.ny, nz=m0.nz, dx=m0.dx, dy=m0.dy, dz=m0.dz)
    if (freqType is None):
        freqType = 'classical'
    freq = g0.initFREQ(freqType)
    freq = torch.from_numpy(freq).to(device)
    
    # laplacian operator freq*freq
    if freqLaplacian == 'modified':
        freqSquare = -g0.initFREQ_laplacian()
    else:
        freqSquare =  freq[:,:,:,0]**2 + freq[:,:,:,1]**2 + freq[:,:,:,2]**2
    
    
    # Green operator
    GG = torch.zeros([g0.nx, g0.ny, g0.nz, 6]).to(device)
    freqSquare[0,0,0] = 1.
    GG[:,:,:,0] = (1. - freq[:,:,:,0]*freq[:,:,:,0]/freqSquare) / (phi0*freqSquare+beta0)
    GG[:,:,:,1] = (1. - freq[:,:,:,1]*freq[:,:,:,1]/freqSquare) / (phi0*freqSquare+beta0)
    GG[:,:,:,2] = (1. - freq[:,:,:,1]*freq[:,:,:,2]/freqSquare) / (phi0*freqSquare+beta0)
    GG[:,:,:,3] = (   - freq[:,:,:,0]*freq[:,:,:,1]/freqSquare) / (phi0*freqSquare+beta0)
    GG[:,:,:,4] = (   - freq[:,:,:,0]*freq[:,:,:,2]/freqSquare) / (phi0*freqSquare+beta0)
    GG[:,:,:,5] = (   - freq[:,:,:,1]*freq[:,:,:,2]/freqSquare) / (phi0*freqSquare+beta0)
    freqSquare[0,0,0] = 0.
    GG[0,0,0,:] = 0.
    
    # allocate variables
    gma     = torch.zeros((m0.nx,m0.ny,m0.nz, 3)).to(device)
    tauF    = torch.zeros((m0.nx,m0.ny,m0.nz, 3), dtype=torch.complex128).to(device)
    vfield  = torch.zeros((m0.nx, m0.ny, m0.nz, 3)).to(device)
    vfieldF = torch.zeros((m0.nx,m0.ny,m0.nz, 3), dtype=torch.complex128).to(device)

    # initialisation: velocity field, laplacian of velocity
    for j0 in range(3):
        # vfield[:,:,:,j0] = 0.
        vfield[:,:,:,j0] = -gmacro[j0]/beta0
            
    # variables for Anderson acceleration
    if p0.cv_acc == True:
        act_R = torch.zeros((m0.nx, m0.ny, m0.nz, 3, p0.AA_depth)).to(device)
        act_U = torch.zeros((m0.nx, m0.ny, m0.nz, 3, p0.AA_depth)).to(device)
    else:
        act_R = None
        act_U = None
        
    if vfield0 is None:
        vfield0 = vfield.clone()
    else:
        vfield = torch.from_numpy(vfield0).to(device).clone()
    
    
    # iterative solution
    iters = 1
    iact = 0
    while True:
        
        # store for Anderson's acceleration
        if p0.cv_acc==True:
            act_U[:,:,:,:,iact] = vfield
            
                
        # Polarisation vectors : store tau in gma
        gma[:,:,:,0] = (phi-phi0)*gma[:,:,:,0] - ((beta[:,:,:,0]-beta0)*vfield[:,:,:,0] + \
                                                   beta[:,:,:,3]*vfield[:,:,:,1] + \
                                                   beta[:,:,:,4]*vfield[:,:,:,2])
        gma[:,:,:,1] = (phi-phi0)*gma[:,:,:,1] - ( beta[:,:,:,3]*vfield[:,:,:,0] + \
                                                  (beta[:,:,:,1]-beta0)*vfield[:,:,:,1] + \
                                                   beta[:,:,:,5]*vfield[:,:,:,2])
        gma[:,:,:,2] = (phi-phi0)*gma[:,:,:,2] - ( beta[:,:,:,4]*vfield[:,:,:,0] + \
                                                   beta[:,:,:,5]*vfield[:,:,:,1] + \
                                                  (beta[:,:,:,2]-beta0)*vfield[:,:,:,2])
        
        # FFT
        for j0 in range(3):
            tauF[:,:,:,j0] = fftn(gma[:,:,:,j0])
        
        
        # apply the green operator -> TODO: calculate GG on the fly
        vfieldF[:,:,:,0] = GG[:,:,:,0]*tauF[:,:,:,0] + \
                           GG[:,:,:,3]*tauF[:,:,:,1] + \
                           GG[:,:,:,4]*tauF[:,:,:,2]
        vfieldF[:,:,:,1] = GG[:,:,:,3]*tauF[:,:,:,0] + \
                           GG[:,:,:,1]*tauF[:,:,:,1] + \
                           GG[:,:,:,5]*tauF[:,:,:,2]
        vfieldF[:,:,:,2] = GG[:,:,:,4]*tauF[:,:,:,0] + \
                           GG[:,:,:,5]*tauF[:,:,:,1] + \
                           GG[:,:,:,2]*tauF[:,:,:,2]
            
        # enforce the loading - macro velocity
        vfieldF[0,0,0,0] = (tauF[0,0,0,0] - gmacro[0]*g0.ntot) / beta0
        vfieldF[0,0,0,1] = (tauF[0,0,0,1] - gmacro[1]*g0.ntot) / beta0
        vfieldF[0,0,0,2] = (tauF[0,0,0,2] - gmacro[2]*g0.ntot) / beta0
        
        # IFFT
        for j0 in range(3):
            vfield[:,:,:,j0] = ifftn(vfieldF[:,:,:,j0]).real
        
        # Anderson's acceleration
        if p0.cv_acc==True:
            act_R[:,:,:,:,iact] = act_U[:,:,:,:,iact]-vfield
            
            # if iters % p0.AA_depth ==0:
            if iters % p0.AA_increment ==0:
                vfield = AAcceleration(act_R, act_U)
                
            if iters % p0.AA_depth == 0:
                iact = 0
            else:
                iact += 1
                
        # macro velocity
        vmacro = torch.zeros(3).to(device)
        vmacro[0] = torch.sum(vfield[:,:,:,0].flatten()) / g0.ntot
        vmacro[1] = torch.sum(vfield[:,:,:,1].flatten()) / g0.ntot
        vmacro[2] = torch.sum(vfield[:,:,:,2].flatten()) / g0.ntot
        
        # convergence check
        if p0.cv_acc==True:
            # if iters % p0.AA_depth == 0:
            if iters % p0.AA_increment == 0:
                pass #Force to do an additional iteration
            else:
                test = torch.sqrt( torch.sum((act_R[:,:,:,0,iact-1]**2 +
                                              act_R[:,:,:,1,iact-1]**2 +
                                              act_R[:,:,:,2,iact-1]**2).flatten())/g0.ntot )
                test = test / torch.sqrt(torch.sum(vmacro**2))
        else:
            test = torch.sqrt( torch.sum(((vfield[:,:,:,0]-vfield0[:,:,:,0])**2 +
                                          (vfield[:,:,:,1]-vfield0[:,:,:,1])**2 +
                                          (vfield[:,:,:,2]-vfield0[:,:,:,2])**2).flatten())/g0.ntot )
            test = test / torch.sqrt(torch.sum(vmacro**2))
        
        
        # pde residual 
        #   R = (phi0*gma_{k} - beta0*vfield_{k} - pfield_{k} + tau_{k-1}) / ||Gmacro||_2
        test_eq = np.nan
        if iters%1==0:
            # Polarisation vector  (at iteration k-1)
            tau = torch.zeros_like(gma)
            for j0 in range(3):
                tau[...,j0] = ifftn(tauF[...,j0])
            
            # Laplacien of velocity (at iteration k)
            for j0 in range(3):
                gma[:,:,:,j0]  = ifftn(-freqSquare* fftn(vfield[:,:,:,j0])).real
            
            # pressure gradient field (at iteration k)
            pfield = torch.zeros_like(vfield)
            pfield[:,:,:,0] = phi*gma[:,:,:,0] - (beta[:,:,:,0]*vfield[:,:,:,0] + \
                                                  beta[:,:,:,3]*vfield[:,:,:,1] + \
                                                  beta[:,:,:,4]*vfield[:,:,:,2])
            pfield[:,:,:,1] = phi*gma[:,:,:,1] - (beta[:,:,:,3]*vfield[:,:,:,0] + \
                                                  beta[:,:,:,1]*vfield[:,:,:,1] + \
                                                  beta[:,:,:,5]*vfield[:,:,:,2])
            pfield[:,:,:,2] = phi*gma[:,:,:,2] - (beta[:,:,:,4]*vfield[:,:,:,0] + \
                                                  beta[:,:,:,5]*vfield[:,:,:,1] + \
                                                  beta[:,:,:,2]*vfield[:,:,:,2])
                                                  
            # equilibrium residual
            eq_residual = phi0*gma - beta0*vfield - pfield + tau
            test_eq = torch.sqrt( torch.sum((eq_residual[:,:,:,0]**2 + \
                                             eq_residual[:,:,:,1]**2 + \
                                             eq_residual[:,:,:,2]**2).flatten())/g0.ntot )
            test_eq = test_eq / torch.sqrt( torch.sum(gmacro**2) )
        
        
        # stop ?
        if test<p0.cv_criterion:
            print('  iteration %d -- residual: U-based (%.3e), E-based (%.3e)'%(iters, test, test_eq))
            break
        
        # counter
        iters += 1
        
        #
        if (iters % 100)==0:
            print('  iteration %d -- residual: U-based (%.3e), E-based (%.3e)'%(iters, test, test_eq))

        # to avoid infinite loop
        if iters>p0.itMax:
            print('  iteration %d -- residual: U-based (%.3e), E-based (%.3e)'%(iters, test, test_eq))
            print('Warning: number of iterations exceeds limit (%d)'%p0.itMax)
            break

        # update
        if p0.cv_acc==False:
            vfield0 = vfield.clone()
            
    
    # macro resistivity
    id_dir = np.argmax( np.abs(vmacro.cpu()) )
    K = -vmacro*mu / gmacro[id_dir]
    print('macro permeability K: ', str(K))
    
    #
    print('residuals: U-based (%.3e),  E-based (%.3e)'%(test, test_eq))
    
    #
    print('Total time (%f)s;  Total nb of iters (%d)'%(time.time()-t, iters))

   
    
    return K.cpu().numpy(), vfield.cpu().numpy(), vmacro.cpu().numpy()








"""
-------------------------------------------------------------------------------
FFT solver for Darcy-Brinkman equation
This is an implementation of the model of Mezhoud, Monchiet, Bonnet, Grande
  see https://doi.org/10.1016/j.euromechflu.2020.04.012
-------------------------------------------------------------------------------

written by Yang Chen, University of Bath, 2022.04.06
re-written by Yang Chen, University of Bath, 2022.12.29
"""
def brinkman_fft_solver_stress( m0,  # microstructure
                                l0,  # load, fluid conditions
                                p0,  # algorithm parameters
                                freqType = None, 
                                freqLaplacian = None,
                                device=None
                                ):
    
    if device is None:
        device = 'cpu'
    
    t = time.time()
    
    # some constant variables
    imP = torch.sqrt(torch.tensor([-1.0], dtype=torch.complex128)).to(device)
            
    # reference material
    phi0  = p0.reference_phi0
    beta0 = p0.reference_beta0
    
    # define distribution funcions
    ks     = m0.micro_permeability  #tensor in global coordinates
    vmacro = torch.as_tensor(l0.macro_load)#macro velocity
    mu     = l0.viscosity           #viscosity in fluid region
    mue    = l0.viscosity_solid     #effective viscosity in porous regions
    
    phi = torch.zeros((m0.nx,m0.ny,m0.nz)).to(device)
    phi[m0.Ifn==0] = mu
    phi[m0.Ifn>0] = mue
    
    # inverse of micro-permeability tensor
    ks_inv = list()
    for i, ks_i in enumerate(ks):
        tmp = np.linalg.inv([[ks_i[0],ks_i[3],ks_i[4]],
                             [ks_i[3],ks_i[1],ks_i[5]],
                             [ks_i[4],ks_i[5],ks_i[2]]])
        ks_inv.append( np.array([tmp[0,0],
                                 tmp[1,1],
                                 tmp[2,2],
                                 tmp[0,1],
                                 tmp[0,2],
                                 tmp[1,2]]) )
        
    #the coefficient beta
    lst_beta = mu * np.array(ks_inv)        #in the local coord syst.
    lst_beta = rot_tensorSym_loc2glob(lst_beta, m0.local_axes[0],m0.local_axes[1]) #in global coord syst.
    
    beta = np.zeros((m0.nx,m0.ny,m0.nz, 6))
    i = 0
    for izone in m0.label_solid:
        IDmatrix = m0.Ifn==izone
        for j0 in range(6):
            beta[IDmatrix,j0] = lst_beta[i,j0]
        i += 1
    beta = torch.from_numpy(beta).to(device)

    # grid and frequency vectors
    g0 = grid(nx=m0.nx, ny=m0.ny, nz=m0.nz, dx=m0.dx, dy=m0.dy, dz=m0.dz)
    if (freqType is None):
        freqType = 'classical'
    freq = g0.initFREQ(freqType)
    freq = torch.from_numpy(freq).to(device)
    
    
    # laplacian operator freq*freq
    if freqLaplacian == 'modified':
        freqSquare = -g0.initFREQ_laplacian()
    else:
        freqSquare = freq[:,:,:,0]**2 + freq[:,:,:,1]**2 + freq[:,:,:,2]**2
    
    # frequency-related tensors (projectors)
    QQ = torch.zeros([g0.nx, g0.ny, g0.nz, 6]).to(device)
    freqSquare[0,0,0] = 1.
    QQ[:,:,:,0] = 1. - freq[:,:,:,0]*freq[:,:,:,0] / freqSquare
    QQ[:,:,:,1] = 1. - freq[:,:,:,1]*freq[:,:,:,1] / freqSquare
    QQ[:,:,:,2] = 1. - freq[:,:,:,2]*freq[:,:,:,2] / freqSquare
    QQ[:,:,:,3] =    - freq[:,:,:,0]*freq[:,:,:,1] / freqSquare
    QQ[:,:,:,4] =    - freq[:,:,:,0]*freq[:,:,:,2] / freqSquare
    QQ[:,:,:,5] =    - freq[:,:,:,1]*freq[:,:,:,2] / freqSquare
    QQ[0,0,0,:] = 0.
    freqSquare[0,0,0] = 0.

    # allocate variables
    omegaF  = torch.zeros((m0.nx,m0.ny,m0.nz, 3), dtype=torch.complex128).to(device)
    sF      = torch.zeros((m0.nx,m0.ny,m0.nz, 6), dtype=torch.complex128).to(device)
    EF      = torch.zeros((m0.nx,m0.ny,m0.nz, 3), dtype=torch.complex128).to(device)
    vfield  = torch.zeros((m0.nx, m0.ny, m0.nz, 3)).to(device)
    vfieldF = torch.zeros((m0.nx,m0.ny,m0.nz, 3), dtype=torch.complex128).to(device)
    strain  = torch.zeros((m0.nx, m0.ny, m0.nz, 6)).to(device)
    
    # initialisation: velocity field, strain field
    for j0 in range(3):
        vfield[:,:,:,j0] = vmacro[j0]
    
    # variables for Anderson acceleration
    if p0.cv_acc == True:
        act_R = torch.zeros((m0.nx, m0.ny, m0.nz, 3, p0.AA_depth)).to(device)
        act_U = torch.zeros((m0.nx, m0.ny, m0.nz, 3, p0.AA_depth)).to(device)
    else:
        act_R = None
        act_U = None
    vfield0 = vfield.clone()
    
    # iterative solution
    iters = 1
    iact = 0
    test = torch.inf
    
    ### Debug only
    lst_tol_U = list()
    lst_tol_E = list()
    ##############
    while True:
        
        # store for Anderson's acceleration
        if p0.cv_acc==True:
            act_U[:,:,:,:,iact] = vfield
            
        # omega, s - quantities related to velocity, strain
        omega   = torch.zeros((m0.nx,m0.ny,m0.nz, 3)).to(device)
        omega[:,:,:,0] = beta[:,:,:,0] * vfield[:,:,:,0] + \
                         beta[:,:,:,3] * vfield[:,:,:,1] + \
                         beta[:,:,:,4] * vfield[:,:,:,2]
        omega[:,:,:,1] = beta[:,:,:,3] * vfield[:,:,:,0] + \
                         beta[:,:,:,1] * vfield[:,:,:,1] + \
                         beta[:,:,:,5] * vfield[:,:,:,2]
        omega[:,:,:,2] = beta[:,:,:,4] * vfield[:,:,:,0] + \
                         beta[:,:,:,5] * vfield[:,:,:,1] + \
                         beta[:,:,:,2] * vfield[:,:,:,2]
            
        
        # FFT of omega (velocity), s (stress)
        for j0 in range(3):
            omegaF[:,:,:,j0] = fftn(omega[:,:,:,j0])
    
        for j00 in range(6):
            sF[:,:,:,j0] = fftn(2.*phi * strain[:,:,:,j0])

        
        # E in Fourier space
        EF = torch.zeros((m0.nx,m0.ny,m0.nz, 3), dtype=torch.complex128).to(device)
        omegaF[:,:,:,0] = imP* (sF[:,:,:,0]*freq[:,:,:,0]+ \
                                sF[:,:,:,3]*freq[:,:,:,1]+ \
                                sF[:,:,:,4]*freq[:,:,:,2]) - omegaF[:,:,:,0]
        omegaF[:,:,:,1] = imP* (sF[:,:,:,3]*freq[:,:,:,0]+ \
                                sF[:,:,:,1]*freq[:,:,:,1]+ \
                                sF[:,:,:,5]*freq[:,:,:,2]) - omegaF[:,:,:,1]
        omegaF[:,:,:,2] = imP* (sF[:,:,:,4]*freq[:,:,:,0]+ \
                                sF[:,:,:,5]*freq[:,:,:,1]+ \
                                sF[:,:,:,2]*freq[:,:,:,2]) - omegaF[:,:,:,2]
        EF[:,:,:,0] = QQ[:,:,:,0]*omegaF[:,:,:,0] + \
                      QQ[:,:,:,3]*omegaF[:,:,:,1] + \
                      QQ[:,:,:,4]*omegaF[:,:,:,2]
        EF[:,:,:,1] = QQ[:,:,:,3]*omegaF[:,:,:,0] + \
                      QQ[:,:,:,1]*omegaF[:,:,:,1] + \
                      QQ[:,:,:,5]*omegaF[:,:,:,2]
        EF[:,:,:,2] = QQ[:,:,:,4]*omegaF[:,:,:,0] + \
                      QQ[:,:,:,5]*omegaF[:,:,:,1] + \
                      QQ[:,:,:,2]*omegaF[:,:,:,2]           
        
        # velocity in Fourier space
        for j0 in range(3):
            vfieldF[:,:,:,j0] = vfieldF[:,:,:,j0] + \
                                EF[:,:,:,j0] / (phi0*freqSquare+beta0)
            vfieldF[0,0,0,j0] = vmacro[j0] * g0.ntot
        
        # strain in Fourier space (store strainF in sF)
        sF[:,:,:,0] = imP* vfieldF[:,:,:,0]*freq[:,:,:,0]
        sF[:,:,:,1] = imP* vfieldF[:,:,:,1]*freq[:,:,:,1]
        sF[:,:,:,2] = imP* vfieldF[:,:,:,2]*freq[:,:,:,2]
        sF[:,:,:,3] = imP*0.5*( vfieldF[:,:,:,0]*freq[:,:,:,1] + \
                                vfieldF[:,:,:,1]*freq[:,:,:,0] )
        sF[:,:,:,4] = imP*0.5*( vfieldF[:,:,:,0]*freq[:,:,:,2] + \
                                vfieldF[:,:,:,2]*freq[:,:,:,0] )
        sF[:,:,:,5] = imP*0.5*( vfieldF[:,:,:,1]*freq[:,:,:,2] + \
                                vfieldF[:,:,:,2]*freq[:,:,:,1] )
            
        # velocity & strain in real space
        for j0 in range(3):
            vfield[:,:,:,j0] = ifftn(vfieldF[:,:,:,j0]).real
        
        for j0 in range(6):
            strain[:,:,:,j0] = ifftn(sF[:,:,:,j0]).real  #strainF stored in sF
        
        
        # Anderson's acceleration
        if p0.cv_acc==True:
            act_R[:,:,:,:,iact] = act_U[:,:,:,:,iact]-vfield
            
            if iters % p0.AA_depth ==0:
                vfield = AAcceleration(act_R, act_U)
                iact = 0
            else:
                iact += 1
        

        # convergence check
        if p0.cv_acc==True:
            if (iters%p0.AA_depth==0):
                pass #Force to do an additional iteration
            else:
                test = torch.sqrt(torch.sum((act_R[:,:,:,0,iact-1]**2+
                                             act_R[:,:,:,1,iact-1]**2+
                                             act_R[:,:,:,2,iact-1]**2).flatten())/g0.ntot)
                test = test / torch.sqrt(torch.sum(vmacro**2))
        else:
            test = torch.sqrt(torch.sum(((vfield[:,:,:,0]-vfield0[:,:,:,0])**2+
                                         (vfield[:,:,:,1]-vfield0[:,:,:,1])**2+
                                         (vfield[:,:,:,2]-vfield0[:,:,:,2])**2).flatten())/g0.ntot)
            test = test / torch.sqrt(torch.sum(vmacro**2))
        
        
        # # macro pressure gradient
        # W = np.zeros(3)
        # W[0] = np.sum( omega[:,:,:,0].flatten() ) / g0.ntot
        # W[1] = np.sum( omega[:,:,:,1].flatten() ) / g0.ntot
        # W[2] = np.sum( omega[:,:,:,2].flatten() ) / g0.ntot
        
        #check the residual of the equilibrium equation
        for j0 in range(3):
            vfield0[:,:,:,j0] = ifftn(EF[:,:,:,j0]).real  #E stored in vfield0
        # resEq = np.sqrt( np.sum((vfield0[:,:,:,0]**2 + \
        #                          vfield0[:,:,:,1]**2 + \
        #                          vfield0[:,:,:,2]**2).flatten())/g0.ntot ) \
        #                 / np.sqrt(np.sum(W**2))
        # # resEq = np.sqrt( np.sum( (vfield0[:,:,:,0]**2 + \
        # #                           vfield0[:,:,:,1]**2 + \
        # #                           vfield0[:,:,:,2]**2).flatten()) ) \
        # #                 / np.sqrt(np.sum(W**2))
        resEq = np.nan
        
        
        ## Debug only
        lst_tol_U.append(test)
        lst_tol_E.append(resEq)
        #############
        
        # stop ?
        if test<p0.cv_criterion:
            break
        
        
        # counter
        iters += 1
        
        #
        if (iters % 500)==0:
            print('  iteration %d -- residual: U-based (%.3e), E-based (%.3e)'%(iters, test, resEq))

            
        # to avoid infinite loop
        if iters>p0.itMax:
            print('Warning: number of iterations exceeds limit (%d)'%p0.itMax)
            break
            
        # update
        vfield0 = np.copy(vfield)
        
        
    # local pressure gradient
    omega = omega * 0.
    omega[:,:,:,0] = beta[:,:,:,0]*vfield[:,:,:,0] + \
                     beta[:,:,:,3]*vfield[:,:,:,1] + \
                     beta[:,:,:,4]*vfield[:,:,:,2]
    omega[:,:,:,1] = beta[:,:,:,3]*vfield[:,:,:,0] + \
                     beta[:,:,:,1]*vfield[:,:,:,1] + \
                     beta[:,:,:,5]*vfield[:,:,:,2]
    omega[:,:,:,2] = beta[:,:,:,4]*vfield[:,:,:,0] + \
                     beta[:,:,:,5]*vfield[:,:,:,1] + \
                     beta[:,:,:,2]*vfield[:,:,:,2]
    
    # macro pressure gradient
    W = torch.zeros(3).to(device)
    W[0] = torch.sum( omega[:,:,:,0].flatten() ) / g0.ntot
    W[1] = torch.sum( omega[:,:,:,1].flatten() ) / g0.ntot
    W[2] = torch.sum( omega[:,:,:,2].flatten() ) / g0.ntot
    
    # macro resistivity
    id_dir = np.argmax( torch.abs(vmacro) )
    H = W / mu / vmacro[id_dir]
    print('macro resistivity H: ', str(H))    
    

    #
    print('residuals: U-based (%.3e),  E-based (%.3e)'%(test, resEq))
    
    #
    print('Total time (%f s);  Total nb of iters (%d)'%(time.time()-t, iters))

    ### Debug only    
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 24}
    plt.rc('font', **font)
    plt.figure()
    plt.semilogy(lst_tol_U, '-', label='based on velocity', linewidth=2)
    plt.semilogy(lst_tol_E, '--', label='based on equilibrium', linewidth=2)
    plt.tick_params(width=1.5, length=5)
    
    # fig,ax=plt.subplots()
    # ax.plot(lst_tol_U, '-', label='based on velocity', linewidth=2)
    # ax.plot(lst_tol_E, '--', label='based on equilibrium', linewidth=2)
    # ax.set_yscale('log')
    # ax.tick_params(width=1.5, length=5)
    # ax.xaxis.label.set_fontsize(36)
    # ax.yaxis.label.set_fontsize(18)
    # plt.show()
    ##############
    
    return H.cpu().numpy(), vfield.cpu().numpy(), -W.cpu().numpy()






