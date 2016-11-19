PathDicom = '/Volumes/Backup Data/ASDH Samples/Sample1/Pre-operative/R-N11-109/HeadSpi  1.0  J40s  3'

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

# Plot heat map
import plotly.plotly as py
py.sign_in('sineadey', 'gb269t15xi')
rot = numpy.rot90(ArrayDicom[:, :, 200])

# using plotly
data = [
    plotly.graph_objs.Heatmap(
        z=rot,
        colorscale='Greys',
    )
]
py.iplot(data, filename='name')

#using pylab
from pylab import *
c = pcolor(rot)
set_cmap('gray')
colorbar()
savefig('plt.png')
axis('equal')
show()

#threshold data with bone
threshold = vtk.vtkImageThreshold ()
threshold.SetInputConnection(reader.GetOutputPort())
threshold.ThresholdByLower(400)  # remove all soft tissue
threshold.ReplaceInOn()
threshold.SetInValue(0)  # set all values below 400 to 0
threshold.ReplaceOutOn()
threshold.SetOutValue(1)  # set all values above 400 to 1
threshold.Update()

imageData = threshold.GetOutput()
pointData = imageData.GetPointData()
assert (pointData.GetNumberOfArrays()==1)
arrayData = pointData.GetArray(0)
_extent = reader.GetDataExtent()
ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
ConstPixelSpacing = reader.GetPixelSpacing()

ArrayDicom = vtk_to_numpy(arrayData)
ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')

# plot again using preferred method