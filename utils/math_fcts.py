# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:44:47 2022
some commonly used functions for basic math operations
@author: yc2634
"""

import numpy as np

"""
some lookup tables for simplifying the symmetric 3x3, 6x6 tensors
"""
def symmetry_index_tables():
    idx3 = np.array([[0,0],
                     [1,1],
                     [2,2],
                     [0,1],
                     [0,2],
                     [1,2]]) #vector form of 3x3 symmetric tensor
    
    idx6 = np.array([[0,0],
                     [1,1],
                     [2,2],
                     [3,3],
                     [4,4],
                     [5,5],
                     [0,1],
                     [0,2],
                     [0,3],
                     [0,4],
                     [0,5],
                     [1,2],
                     [1,3],
                     [1,4],
                     [1,5],
                     [2,3],
                     [2,4],
                     [2,5],
                     [3,4],
                     [3,5],
                     [4,5]]) #vector form of 6x6 symmetric tensor
    
    idx3t = np.array([[0,3,4],
                      [3,1,5],
                      [4,5,2]])
    
    idx6t = np.array([[0,   6,  7,  8,  9, 10],
                      [6,   1, 11, 12, 13, 14],
                      [7,  11,  2, 15, 16, 17],
                      [8,  12, 15,  3, 18, 19],
                      [9,  13, 16, 18,  4, 20],
                      [10, 14, 17, 19, 20, 5]])

    return idx3, idx6, idx3t, idx6t




"""
generic form for the lookup tables to simpfify symmetric nxn tensors
    Input:
        n - dimension of the symmetric tensor (nxn)
            number of independent elements is m=(n+1)*n/2
    Output:
        idx_tens2vec - nxn array, such that
                           j00 = idx_tens2vec[j0,j1]
                           vectorForm[j00] = tensorForm[j0,j1]
        idx_vec2tens - mx1 nested array, such that 
                           j0, j1 = idx_vec2tens[j00]
                           tensorForm[j0,j1] = vectorForm[j00]
"""
def symmetry_index_tables_generic(n):
    
    idx_vec2tens = list()
    # diagonal elements
    for j0 in range(n):
        idx_vec2tens.append( [j0, j0] )
    # non-diagonal elements
    for j0 in range(n):
        for j1 in range(j0+1,n):
            idx_vec2tens.append( [j0, j1] )
    idx_vec2tens = np.array(idx_vec2tens)

    idx_tens2vec = np.zeros((n,n),dtype=int)
    # diagonal elements
    for j0 in range(n):
        idx_tens2vec[j0,j0] = j0
    # non-diagonal elements
    i = n-1
    for j0 in range(n):
        for j1 in range(j0+1,n):
            i += 1
            idx_tens2vec[j0,j1] = i
            idx_tens2vec[j1,j0] = i
            
    return idx_tens2vec, idx_vec2tens




"""
Rotation of a symmetric 3x3 matrix from the local coords to global coords
    Inputs:
         A : vector form of the symmetric 3x3 matrix to be rotated
               nx6 matrix, with n the number of points
        ex : unit vector for the x-axis in the local coodinate system
               nx3 matrix with n the number of points
        ey : unit vector for the y-axis in the local coodinate system
               nx3 matrix
               
    Outputs:
         B : rotated matrix (in vector form)
               nx6 matrix
"""
def rot_tensorSym_loc2glob(A, ex, ey):
    #
    num = np.shape(A)[0]
    
    ez      = np.zeros((num, 3))
    ez[:,0] = ex[:,1]*ey[:,2] - ex[:,2]*ey[:,1]
    ez[:,1] = ex[:,2]*ey[:,0] - ex[:,0]*ey[:,2]
    ez[:,2] = ex[:,0]*ey[:,1] - ex[:,1]*ey[:,0]
    
    Q = np.zeros((num,3,3))
    Q[:,0,0] = ex[:,0]
    Q[:,1,0] = ex[:,1]
    Q[:,2,0] = ex[:,2]
    Q[:,0,1] = ey[:,0]
    Q[:,1,1] = ey[:,1]
    Q[:,2,1] = ey[:,2]
    Q[:,0,2] = ez[:,0]
    Q[:,1,2] = ez[:,1]
    Q[:,2,2] = ez[:,2]
    
    idx3, _, idx3t, _ = symmetry_index_tables()
    
    B = np.zeros((num,6))
    for j00 in range(6):
        j0, j1 = idx3[j00]
        for j2 in range(3):
            for j3 in range(3):
                B[:,j00] += Q[:,j0,j2]*A[:,idx3t[j2,j3]]*Q[:,j1,j3]
                
    return B


"""
Anderson's acceleration
    Inputs:
        act_R : residual vector
        act_U : unknown vector
    Output:
            U : "accelerated" unknown 
"""
def AAcceleration(act_R, act_U):
    act_R = np.moveaxis(act_R, -1, 0)
    act_U = np.moveaxis(act_U, -1, 0)
    
    nd = act_R.shape[0]
    
    A = np.zeros((nd,nd)) #TODO: improve this with symmetry!!!
    for j0 in range(nd):
        for j1 in range(nd):
            tmp = np.sum( (act_R[j0]*act_R[j1]).flatten() )
            A[j0,j1] = tmp
            A[j1,j0] = tmp
    Ainv = np.linalg.pinv(A)
    
    s = np.sum(Ainv,axis=0) / np.sum(Ainv.flatten())
    
    U = np.zeros(act_U.shape[1:], dtype=act_U.dtype)
    for i0 in range(nd):
        U += s[i0] * (act_U[i0]-act_R[i0])

    return U

def AAcceleration_old(act_R, act_U):
    nx,ny,nz,nc,nd = np.shape(act_R)
    
    A = np.zeros((nd,nd)) #TODO: improve this with symmetry!!!
    for j0 in range(nd):
        for j1 in range(nd):
            tmp = np.sum( (act_R[:,:,:,:,j0]*act_R[:,:,:,:,j1]).flatten() )
            A[j0,j1] = tmp
            A[j1,j0] = tmp
    Ainv = np.linalg.pinv(A)
    
    s = np.sum(Ainv,axis=0) / np.sum(Ainv.flatten())
    
    U = np.zeros((nx,ny,nz,nc), dtype=act_U.dtype)
    for i0 in range(nd):
        U += s[i0] * (act_U[:,:,:,:,i0]-act_R[:,:,:,:,i0])

    return U


"""
calculate the inverse matrix of a symmetric 3x3 matrix
  with the input matrix expressed in a column vector
  hence the output will also be expressed in a column vector
    Inputs:
           ks : input symmetric 3x3 matrix expressed in nx6 column vector
    Outputs:
       ks_inv : output symetric 3x3 matrix expressed in nx6 column vector
"""
def inv_matrix3x3sym_vec(ks):
    
    ks_inv = np.zeros(ks.shape, dtype=ks.dtype)
    
    for i, ks_i in enumerate(ks):
        ks_i_t = np.array([[ks_i[0],ks_i[3],ks_i[4]],
                           [ks_i[3],ks_i[1],ks_i[5]],
                           [ks_i[4],ks_i[5],ks_i[2]]])
        tmp = np.linalg.inv(ks_i_t)
        ks_inv[i,:] = [tmp[0,0], tmp[1,1], tmp[2,2], tmp[0,1], tmp[0,2], tmp[1,2]]
    
    return ks_inv
    
