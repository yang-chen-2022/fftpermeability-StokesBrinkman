# fftpermeability-StokesBrinkman
FFT solvers for predicting the permeability of porous media

* Stokes-Brinkman problem
* Fixed-point iterative algorithm
* Anderson acceleration

* New implementation with PyTorch for GPU acceleration. Even witout GPU, the torch implementation performs better with CPU parallisation.
  
# how to run examples
    make sure you are in the root directory, then type
    python -m examples.parallelChannel

 
## Please cite:
  Yang Chen, "High-performance computational homogenization of Stokes-Brinkman flow with an Anderson-accelerated FFT method.", Iinternational Journal of Numerical Methods in Fluids. https://doi.org/10.1002/fld.5199
  
### A Fortran implementation is available upon request. 
