import numpy as np
from scipy.fft import fftfreq, fftn

# classes

class grid:
    
    def __init__(self, nx=1, ny=1, nz=1, dx=1, dy=1, dz=1): #constructor
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        
        self.T1 = self.nx * self.dx 
        self.T2 = self.ny * self.dy 
        self.T3 = self.nz * self.dz
        
        self.ntot = self.nx * self.ny * self.nz
    

    def initFREQ(self, choice):

        if choice=='classical':
            DF1 = 2. * np.pi / self.T1
            DF2 = 2. * np.pi / self.T2
            DF3 = 2. * np.pi / self.T3
            
            freq = np.zeros((self.nx, self.ny, self.nz, 3))
            
            k = DF1 * fftfreq(self.nx,1./self.nx)
            for i in range(self.nx):
                freq[i,:,:,0] = k[i]
                
            k = DF2 * fftfreq(self.ny,1./self.ny)
            for i in range(self.ny):
                freq[:,i,:,1] = k[i]
                
            k = DF3 * fftfreq(self.nz,1./self.nz)
            for i in range(self.nz):
                freq[:,:,i,2] = k[i]
            
        elif choice=='modified':
            
            #note: this implementation is identical to AMITEX for filter_radius=1 (hexa)
            ii = np.pi * fftfreq(self.nx,1./self.nx) / self.nx
            jj = np.pi * fftfreq(self.ny,1./self.ny) / self.ny
            kk = np.pi * fftfreq(self.nz,1./self.nz) / self.nz
            
            jj,ii,kk = np.meshgrid(jj, ii, kk)
            
            freq = np.zeros((self.nx, self.ny, self.nz, 3))
            freq[:,:,:,0] = 2./self.dx * np.sin(ii)*np.cos(jj)*np.cos(kk)
            freq[:,:,:,1] = 2./self.dy * np.cos(ii)*np.sin(jj)*np.cos(kk)
            freq[:,:,:,2] = 2./self.dz * np.cos(ii)*np.cos(jj)*np.sin(kk)

        return freq


    def initFREQ_laplacian(self):

        ii = 2.*np.pi * fftfreq(self.nx,1./self.nx) / self.nx
        jj = 2.*np.pi * fftfreq(self.ny,1./self.ny) / self.ny
        kk = 2.*np.pi * fftfreq(self.nz,1./self.nz) / self.nz
        
        jj,ii,kk = np.meshgrid(jj, ii, kk)

        freqLaplacian = 2. * (np.cos(ii)-1.) / self.dx**2 \
                      + 2. * (np.cos(jj)-1.) / self.dy**2 \
                      + 2. * (np.cos(kk)-1.) / self.dz**2
        
        return freqLaplacian




class microstructure:
    
    def __init__(self,          Ifn , 
                                  L , 
                       label_solid=1,
                       label_fluid=0,
                       label_B=None,
                       micro_permeability=None,
                       local_axes=None):
        
        self.Ifn         = Ifn #characteristic function
        self.L           = L
        self.label_solid = label_solid
        self.label_fluid = label_fluid
        if label_B !=None:
            self.label_B = label_B
        self.micro_permeability = micro_permeability
        self.local_axes = local_axes
        
        self.nx, self.ny, self.nz = np.shape(Ifn)
        self.dx = L[0] / self.nx
        self.dy = L[1] / self.ny
        self.dz = L[2] / self.nz

    def fft_charact_fct(self): # fft of the characteristic function
        return fftn(self.Ifn)
    
    def vol_frac_solid(self):
        return np.count_nonzero(self.Ifn==self.label_solid) / np.size(self.Ifn)


class load_fluid_condition:
    
    def __init__(self,      macro_load = [1.,0.,0.],
                             viscosity = 1., 
                       viscosity_solid = None ):
        
        self.macro_load       = macro_load 
        self.viscosity        = viscosity
        self.viscosity_solid  = viscosity_solid
        


class param_algo:
    
    def __init__(self,
                 cv_criterion  = 1e-4,
                 reference_mu0 = 0.5,
                reference_phi0 = None,
               reference_beta0 = None,
                  reference_ks = 0.,
                velocity_scale = 1.,
                         itMax = 1000,
                        cv_acc = False,
                      AA_depth = 4,
                  AA_increment = 4
                      ):
        self.cv_criterion    = cv_criterion
        self.reference_mu0   = reference_mu0
        self.itMax           = itMax
        self.reference_phi0  = reference_phi0
        self.reference_beta0 = reference_beta0
        self.reference_ks    = reference_ks
        self.velocity_scale  = velocity_scale
        self.cv_acc          = cv_acc
        if cv_acc == True:
            self.AA_depth = AA_depth
            self.AA_increment = AA_increment
        

if __name__ == "__main__":
    
    #check the grid class and its methods
    g0 = grid()
    freq = g0.initFREQ()
    freq_rotated = g0.initFREQ_rotated()
    
    #check the param_algo class and its methods
    p0 = param_algo()
    
    #check the microstructure class and its methods
    m0 = microstructure(np.zeros([3,5,7]), [4.,4.,1.])
    IfnF = m0.fft_charact_fct()
    cs = m0.vol_frac_solid()
    
    
