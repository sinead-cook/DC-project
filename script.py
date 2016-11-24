import vtk
from vtk.util.numpy_support import vtk_to_numpy
import plotly
from plotly.graph_objs import *
import numpy as np
import cv2
import matplotlib

def cv2sift(ArrayDicom, depth):
    image_array=ArrayDicom[:,:,depth]
    matplotlib.image.imsave('sift1post.png', image_array)
    img = cv2.imread('sift1post.png')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,img)
    cv2.imwrite('sift2post.png',img)

def cv2siftmatch(comparison_slice, array):
    matplotlib.image.imsave('comparison_slice.png',comparison_slice)
    img = cv2.imread('comparison_slice.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,img)
    cv2.imwrite('comparison_slice_features.png',img)
        

def dicom2np(PathDicom):   
    #PathDicom = '/Volumes/Backup Data/ASDH Samples/Sample1/Pre-operative/R-N11-109/HeadSpi  1.0  J40s  3'
    
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
    
def vtk2np(input, PathDicom):
    pointData = input.GetPointData()
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
    return(ArrayDicom)

def heatmap(ArrayDicom, depth):
    # Plot heat map
    import numpy
    py.sign_in('sineadey', 'gb269t15xi')
    rot = numpy.rot90(ArrayDicom[:, :, depth])
    
    # using plotly
    data = [
        py.graph_objs.Heatmap(
            z=rot,
            colorscale='Greys',
        )
    ]
    plotly.offline.iplot(data, filename='name')
    
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
    
def threshim(low, high, PathDicom):
    # http://www.programcreek.com/python/example/65395/vtk.vtkThreshold
    
    #PathDicom = '/Volumes/Backup Data/ASDH Samples/Sample1/Pre-operative/R-N11-109/HeadSpi  1.0  J40s  3'

    #threshold data with bone
    import vtk
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(PathDicom)
    reader.Update()
    threshold = vtk.vtkImageThreshold ()
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.ThresholdByLower(low)  # remove all soft tissue
    threshold.ReplaceInOn()
    threshold.SetInValue(0)  # set all values below 400 to 0
    threshold.ReplaceOutOn()
    threshold.SetOutValue(1)  # set all values above 400 to 1
    threshold.Update()
    
    if high != 'none':
        threshold.ThresholdByLower(high)  # remove all soft tissue
        threshold.ReplaceOutOn()
        threshold.SetOutValue(0)  # set all values above high to 0
        threshold.Update()
    return threshold.GetOutput()
    
def visualise_array(array,depth):
    import PIL
    import numpy as np
    from PIL import Image
    im = Image.fromarray(array[:, :, depth])
    im1 = im.convert('L')
    return im1
    
def thresh_and_visualise(low, high, PathDicom, depth):
    thresholded=threshim(low,high,PathDicom)
    array=vtk2np(thresholded,PathDicom)
    im=visualise_array(array,depth)
    return im

#PathDicom = '/Volumes/Backup Data/ASDH Samples/Sample1/Pre-operative/R-N11-109/HeadSpi  1.0  J40s  3'
#
#imageData = threshim(400, 'none', PathDicom)
#pointData = imageData.GetPointData()
#assert (pointData.GetNumberOfArrays()==1)
#arrayData = pointData.GetArray(0)
#_extent = reader.GetDataExtent()
#ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
#ConstPixelSpacing = reader.GetPixelSpacing()
#
#ArrayDicom = vtk_to_numpy(arrayData)
#ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')

# plot again using preferred method