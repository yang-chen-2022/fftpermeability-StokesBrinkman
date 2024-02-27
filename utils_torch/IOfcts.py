import vtk
from vtk.util import numpy_support
from vtk.util.numpy_support import numpy_to_vtk

import numpy as np

"""
 #file name *.vti
#volume data (3D, 4D array, shape: [x,y,z] or [x,y,z,c])
#volume name, e.g. velocity, phase, etc.
#origin
#spacing in x,y,z direction, (3x1 array)
#number of voxels in x,y,z direction, (3x1 array)
"""
def saveField2VTK(fileout, vdata, vname, origin=[0,0,0], spacing=[1,1,1]):

    #
    dx, dy, dz = spacing
    x0, y0, z0 = origin
    vcomponents = np.array(vdata[0,0,0]).size

    #dimension + swap components in case of vector input    
    if vcomponents==1:
        nx, ny, nz = np.shape(vdata)
    elif vcomponents==3:
        nx, ny, nz = np.shape(vdata[:,:,:,0])
        # vdata1 = np.zeros(np.shape(vdata))
        # vdata1[:,:,:,0] = vdata[:,:,:,1]
        # vdata1[:,:,:,1] = vdata[:,:,:,0]
        # vdata1[:,:,:,2] = vdata[:,:,:,2]
        # tmp = np.copy(vdata[:,:,:,0])
        # vdata[:,:,:,0] = np.copy(vdata[:,:,:,2])
        # vdata[:,:,:,2] = np.copy(tmp)

    #swap x and z axes
    vdata = np.swapaxes(vdata, 0, 2)

    #data type
    vtype = vtk.util.numpy_support.get_vtk_array_type(vdata.dtype)
    
    #create vtk image object
    imageData = vtk.vtkImageData()
    imageData.SetSpacing(dx, dy, dz)
    imageData.SetOrigin(x0, y0, z0)
    imageData.SetDimensions(nx, ny, nz)
    imageData.AllocateScalars(vtype, vcomponents)
    
    vtk_data_array = numpy_to_vtk(num_array=vdata.ravel(), deep=True, array_type=vtype)
    vtk_data_array.SetNumberOfComponents(vcomponents)
    vtk_data_array.SetName(vname)
    imageData.GetPointData().SetScalars(vtk_data_array)
    
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(imageData)
    writer.SetFileName(fileout)
    writer.Write()
