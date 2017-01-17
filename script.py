# conda install mayavi
# conda install plotly
# pip install git+https://www.github.com/hbldh/b2ac for the ellipse function

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode()
import numpy as np
# import cv2
import matplotlib
from IPython.display import Image
import nibabel as nib

def dicom2np():
    
    
    from __main__ import PathDicom
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(PathDicom)
    reader.Update()
    
    # Get the 'vtkImageData' object from the reader
    imageData = reader.GetOutput()
    # Get the 'vtkPointData' object from the 'vtkImageData' object
    pointData = imageData.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays()==1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)
    
    # Load dimensions using `GetDataExtent`
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    
    # Load spacing values
    ConstPixelSpacing = reader.GetPixelSpacing()
    
    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    return(ArrayDicom)

def nii2np():
    # http://localhost:8888/edit/DC-project/script.py for nii
    from __main__ import PathDicom
    img = nib.load(PathDicom)
    img_data = img.get_data()
    return(img_data)
    
def vtk2np(input_):
    from __main__ import PathDicom
    pointData = input_.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays()==1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)
    
    # Load dimensions using `GetDataExtent`
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(PathDicom)
    reader.Update()
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    
    # Load spacing values
    ConstPixelSpacing = reader.GetPixelSpacing()
    
    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    return(ArrayDicom, ConstPixelSpacing)

def vtk2np_nii(input_):
    from __main__ import PathDicom
    pointData = input_.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays()==1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)
    
    # Load dimensions using `GetDataExtent`
    from __main__ import PathDicom
    img = nib.load(PathDicom)
    img_data = img.get_data()
    img_data_shape = img_data.shape
    _extent = (0, img_data_shape[0] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[2] - 1)
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    header = img.header
    spacing = header['pixdim'][1:4]
    # Load spacing values
    ConstPixelSpacing = spacing[0], spacing[1], spacing[2]
    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    return(ArrayDicom, ConstPixelSpacing)

def heatmap(ArrayDicom, depth, x, y):

    heatmap = go.Heatmap(
            x = x,
            y = y,
            z = ArrayDicom[:,:,depth],
            colorscale = 'Greys'
            )

    layout = go.Layout(
        width = 600,
        height= 600,
        title='Slice number %i' % depth
    )
    data = [heatmap]
    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)


    
def pyheatmap(ArrayDicom, slice_, axis):
    if axis==1:
        rot = np.rot90(ArrayDicom[slice_, :, :])
    elif axis==2:
        rot = np.rot90(ArrayDicom[:, slice_, :])
    elif axis==3:
        rot = np.rot90(ArrayDicom[:, :, slice_])
    else:
        print 'axis number needs to be 1, 2 or 3'
    #using pylab
    from pylab import *
    c = pcolor(rot)
    set_cmap('gray')
    colorbar()
    savefig('plt.png')
    #axis('equal')
    show()
    
def threshim_dicom(low, high):
    # http://www.programcreek.com/python/example/65395/vtk.vtkThreshold
    from __main__ import PathDicom
    #threshold data with bone
    import vtk
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(PathDicom)
    reader.Update()
    threshold = vtk.vtkImageThreshold ()
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.ThresholdByLower(low)  # remove all soft tissue
    threshold.ReplaceInOn()
    threshold.SetInValue(0)  # set all values below low to 0
    #threshold.ReplaceOutOn()
    #threshold.SetOutValue(1)  # set all values above 400 to 1
    threshold.Update()

    if high != 'none':
        threshold2 = vtk.vtkImageThreshold ()
        threshold2.SetInputConnection(threshold.GetOutputPort())
        threshold2.ThresholdByUpper(high)  # remove all soft tissue
        threshold2.Update()
        threshold2.ReplaceInOn()
        threshold2.SetInValue(0)  # set all values above high to 0
        threshold2.Update()
        thresholded = threshold2.GetOutput()
    else:
        thresholded = threshold.GetOutput()
    return thresholded

def threshim_nii(low, high):
    # http://www.programcreek.com/python/example/65395/vtk.vtkThreshold
    from __main__ import PathDicom
    #threshold data with bone
    img = nib.load(PathDicom)
    img_data = img.get_data()
    img_data_shape = img_data.shape
    header = img.header

    spacing = header['pixdim'][1:4]

    dataImporter = vtk.vtkImageImport()
    dataImporter.SetDataScalarTypeToShort()
    data_string = img_data.flatten('F').tostring()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataExtent(0, img_data_shape[0] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[2] - 1)
    dataImporter.SetWholeExtent(0, img_data_shape[0] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[2] - 1)
    dataImporter.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    dataImporter.Update()
        
    threshold = vtk.vtkImageThreshold ()
    threshold.SetInputConnection(dataImporter.GetOutputPort())
    threshold.ThresholdByLower(low)  # remove all soft tissue
    threshold.ReplaceInOn()
    threshold.SetInValue(0)  # set all values below low to 0
    #threshold.ReplaceOutOn()
    #threshold.SetOutValue(1)  # set all values above 400 to 1
    threshold.Update()

    if high != 'none':
        threshold2 = vtk.vtkImageThreshold ()
        threshold2.SetInputConnection(threshold.GetOutputPort())
        threshold2.ThresholdByUpper(high)  # remove all soft tissue
        threshold2.Update()
        threshold2.ReplaceInOn()
        threshold2.SetInValue(0)  # set all values above high to 0
        threshold2.Update()
        thresholded = threshold2.GetOutput()
    else:
        thresholded = threshold.GetOutput()
    return thresholded

def orientation(numpy_array): #numpy_array needs to be binary/thresholded
    import b2ac.preprocess
    import b2ac.fit
    import b2ac.conversion
    for i in range(numpy_array.shape[0]):
        for j in range(numpy_array.shape[1]):
            if abs(numpy_array[i,j])<0.01:
                numpy_array[i,j] = 0
    indices = np.nonzero(numpy_array) # will return the indices of any nonzero values
    x = indices[1]
    y = indices[0]
    points = np.zeros((len(indices[1]),2))
    for i in range(len(indices[1])):
        points[i, 0] = x[i]
        points[i, 1] = y[i]
    try:
        points, x_mean, y_mean = b2ac.preprocess.remove_mean_values(points)
        # Fit using NumPy methods in double precision.

        conic_double = b2ac.fit.fit_improved_B2AC_double(points)

        # Convert from conic coefficient form to general ellipse form.
        general_form_double = b2ac.conversion.conic_to_general_1(conic_double)
        general_form_double[0][0] += x_mean
        general_form_double[0][1] += y_mean
    except:
        general_form_double = 0
    return general_form_double # [x, y], [x_axis, y_axis], angle
    
def hist_control(data): # optimise number of bins to get the lowest threshold fraction. A thresholdfraction of 0.1 means that the bin with the 2nd highest frequency will have 0.1xthe frequency of the first highest bin.
    # (1) get rid of frequencies outside a standard deviation of 1. 
    """ 
    @ return: Return array. 1st number is the number of bins, 2nd number is the fraction achieved (where fraction is the 2nd highest frequency divided by the highest frequency, 3rd number is the midbin value of the bin corresponding to the highest frequency. 
    """
    from scipy.stats import norm
    hist, bins= np.histogram(data, bins=20)
    mu,sd = norm.fit(data)
    for i in range(len(hist)):
        if bins[i] < (mu-sd) or bins[i] > (mu+sd):
            hist[i] = 0
    # (2) optimise number of bins to get largest differences between 1st and second highest frequences
    fractions = np.zeros((50,3))
    for num_bins in range(10,60):
        hist, bins = np.histogram(data, bins=num_bins)
        midbins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        data = np.transpose(np.array([hist,midbins]))
        sorted_ = data[data[:,0].argsort()]
        fractions[num_bins-10][0]=num_bins # number of bins 
        fractions[num_bins-10][1]=sorted_[-2][0]/sorted_[-1][0] # fraction 
        fractions[num_bins-10][2]=sorted_[-1][1] # midbin value
    ind = np.where(fractions[1]==min(fractions[1]))[0][0]
    return fractions[ind]

def reshape(array, ConstPixelSpacing, a, c, width, depth):
    """ Reshape array to a 220 by 220 by 160 array with uniform spacing """
    reshaped_array1 = np.zeros((a,width,depth))

    xp = np.linspace(0, ConstPixelSpacing[0]*width, width)
    x = np.linspace(0, a-1, a)

    for j in range(width): # middle dimension
        for k in range(depth): # last dimension
            reshaped_array1[:,j,k] = np.interp(x, xp, array[:,j,k])

    reshaped_array2 = np.zeros((a,a,depth))

    for i in range(a): # first dimension
        for k in range(depth): # last dimension
            reshaped_array2[i,:,k] = np.interp(x, xp, reshaped_array1[i,:,k])

    reshaped_array3 = np.zeros((a,a,c))
    xp = np.linspace(0, ConstPixelSpacing[2]*depth, depth)
    x = np.linspace(0, c-1, c)

    for i in range(a): # first dimension
        for j in range(a): # middle dimension
            reshaped_array3[i,j,:] = np.interp(x, xp, reshaped_array2[i,j,:])

    return reshaped_array3