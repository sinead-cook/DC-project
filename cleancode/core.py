import numpy as np
import nibabel as nib
import os

def rejectOutliers(data, m=2):
    dx = np.abs(data[0] - np.median(data[0]))
    dy = np.abs(data[1] - np.median(data[1]))
    dz = np.abs(data[2] - np.median(data[2]))
    mdevx = np.median(dx)
    mdevy = np.median(dy)
    mdevz = np.median(dz)
    # if else is preventing from dividing by 0. if the median is 0, all
    # the differences have to be 0 becaues dx, dy, dz don't contain
    # negative numbers
    sx = dx/mdevx if mdevx else 0. 
    sy = dy/mdevy if mdevy else 0.
    sz = dz/mdevz if mdevz else 0.
    logicalx = sx<m
    logicaly = sy<m
    logicalz = sz<m
    e = data*logicalx*logicaly*logicalz
    data_return = np.array(e[:,e[1]!=0])
    return data_return

def dicom2np(path):
    """ 
    Converts a dicom file to a numpy array. 
    Returns img_data, the numpy array and ConstPixelSpacing, 
    a list of the spacing between datapoints in mm for each dimension
    """
    pathDicom = os.path.dirname(path)

    import dicom
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(pathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    # Get ref files
    RefDsFirst = dicom.read_file(lstFilesDCM[0])
    RefDsLast  = dicom.read_file(lstFilesDCM[-1])
    ImagePositionPatientFirst = RefDsFirst.ImagePositionPatient
    ImagePositionPatientLast  = RefDsLast.ImagePositionPatient
    # Load dimensions based on the number of rows, columns, 
    # and slices (along the Z axis)
    ConstPixelDims = (int(RefDsFirst.Rows), int(RefDsFirst.Columns), len(lstFilesDCM))
    
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDsFirst.PixelSpacing[0]), float(RefDsFirst.PixelSpacing[1]), 
        float(RefDsFirst.SliceThickness))
    
    # The array is sized based on 'ConstPixelDims'
    array = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
    
    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the files
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        array[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  
        
    T1 = np.array(ImagePositionPatientFirst).astype(float)
    TN = np.array(ImagePositionPatientLast).astype(float)
    thirdCol = np.divide(T1-TN,1.-float(array.shape[2]))

    F1 = np.array(RefDsFirst.ImageOrientationPatient[:3]).astype(float)
    F2 = np.array(RefDsFirst.ImageOrientationPatient[3:6]).astype(float)
    affine = np.zeros((4,4))
    affine[:,1][:3] = -F1*ConstPixelSpacing[0]
    affine[:,0][:3] = -F2*ConstPixelSpacing[1]
    affine[:,2][:3] = thirdCol
    affine[:,3][:3] = [-T1[0], -T1[1], T1[2]]
    affine[3,3] = 1.    
    
    return(array, ConstPixelSpacing, affine)

def nifti2np(pathNifti):
    """ 
    Converts a NIFTI file to a numpy array. 
    Returns img_data, the numpy array and ConstPixelSpacing, 
    a list of the spacing between datapoints in mm for each dimension
    """
    img = nib.load(pathNifti)
    img_data = img.get_data()
    header = img.header
    img_data_shape = img_data.shape
    _extent = (0, img_data_shape[0] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[2] - 1)
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    affine = img.header.get_best_affine()
    spacing = header['pixdim'][1:4]
    # Load spacing values
    ConstPixelSpacing = spacing[0], spacing[1], spacing[2]
    return(img_data, ConstPixelSpacing, affine)

def nrrd2np(pathNrrd):
    """ 
    Converts a nrrd file to a numpy array. 
    Returns img_data, the numpy array and ConstPixelSpacing, 
    a list of the spacing between datapoints in mm for each dimension
    """
    import nrrd
    readdata, options = nrrd.read(pathNrrd)
    ConstPixelSpacing = []
    spaceDirections = np.array(options['space directions'])
    ImagePositionPatient = options['space origin']
    for k in range(len(spaceDirections)):
        ConstPixelSpacing.append(np.linalg.norm(spaceDirections[k]))
    spaceDir = np.vstack(options['space directions'])
    spaceOr = options['space origin']
    affine = np.zeros((4,4))
    affine[:3,:3] = spaceDir
    affine[:3, 3] = spaceOr
    affine[3,3] = .1

    affine[3,3] = affine[3,3]*-1
    affine[0:2,0:2] = -affine[0:2,0:2]
    affine[0:2,3] = -affine[0:2,3]

    return(readdata, ConstPixelSpacing, affine)


def thresholdnp(array, lo, hi):
    thresholded1 = np.multiply(array, (array>lo).astype(int))
    thresholded2 = np.multiply(thresholded1, (array<hi).astype(int))
    return thresholded2

def orientation(numpy_array): #numpy_array needs to be binary/thresholded
    import b2ac.preprocess
    import b2ac.fit
    import b2ac.conversion
    numpy_array[abs(numpy_array)<0.01] = 0
    numpy_array[~np.isfinite(numpy_array)] = 0
    indices = np.nonzero(numpy_array)
    if len(indices[0])>1:
        x = indices[1]
        y = indices[0]
        z = indices[0] - indices[0]
        xyz = rejectOutliers(np.array([x,y,z]))
        x = xyz[0, :]
        y = xyz[1, :]
        points = np.zeros((len(indices[1]),2))
        for i in range(x.shape[0]):
            points[i, 0] = x[i]
            points[i, 1] = y[i]
        if points.shape[0] > 1:
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
        else: general_form_double = 0
    else: general_form_double = 0
    return general_form_double # [x, y], [x_axis, y_axis], angle
    
    
def reshape(array, ConstPixelSpacing):
    """ Reshape array to have cubic voxels of size 1mm^3 """

    width = array.shape[0]
    height = array.shape[1]
    depth = array.shape[2]

    a = int(width*ConstPixelSpacing[0])
    b = int(height*ConstPixelSpacing[1])
    c = int(depth*ConstPixelSpacing[2])

    reshapedArray1 = np.zeros((a,height,depth))

    xp = np.linspace(0, width*ConstPixelSpacing[0], width)
    x  = np.linspace(0, a-1, a)

    for j in range(height): # middle dimension
        for k in range(depth): # last dimension
            reshapedArray1[:,j,k] = np.interp(x, xp, array[0:len(xp),j,k])

    reshapedArray2 = np.zeros((a,b,depth))
    yp = np.linspace(0, height*ConstPixelSpacing[1], height)
    y  = np.linspace(0, b-1, b)

    for i in range(a): # first dimension
        for k in range(depth): # last dimension
            reshapedArray2[i,:,k] = np.interp(y, yp, reshapedArray1[i,0:len(yp),k])

    reshapedArray3 = np.zeros((a,b,c))
    zp = np.linspace(0, depth*ConstPixelSpacing[2], depth)
    z = np.linspace(0, c-1, c)

    for i in range(a): # first dimension
        for j in range(b): # middle dimension
            reshapedArray3[i,j,:] = np.interp(z, zp, reshapedArray2[i,j,0:len(zp)])

    return reshapedArray3

def ellipses(bone):
    angles             =np.zeros(bone.shape[2])
    xcentroids         =np.zeros(bone.shape[2])
    ycentroids         =np.zeros(bone.shape[2])
    xmajor_axis_points =np.zeros(bone.shape[2])
    ymajor_axis_points =np.zeros(bone.shape[2])

    for i in range(bone.shape[2]):
    #    print 'Slice number {}'.format(i)
        array_i           = bone[:,:,i] # array to pass to 'orientation' must be thresholded already
        orientation_i     = orientation(array_i)
        if orientation_i != 0: 
            angles[i]      =orientation_i[2] # pick out the angle
            xcentroids[i]  =orientation_i[0][0]
            ycentroids[i]  =orientation_i[0][1]
            
        else: pass
    #    else: print 'Slice {} contains 0'.format(i)
    return angles, xcentroids, ycentroids

def selectEllipsesRange(angles):
    # make all angles positive
    angles[angles < 0] += np.pi
    import scipy.signal as signal
    # First, design the Butterworth filter
    N  = 1   # Filter order - the higher the order, the sharper the dropoff
    pd = float(20)  # Cutoff period - the inverse is the cutoff frequency
    fs = 1 # Sample rate frequency
    nyq = 0.5*fs
    low = 1/pd
    wn = low/nyq # Cutoff frequency as a fraction
    b, a = signal.butter(N, wn, output='ba')
    filt_angles = signal.filtfilt(b,a, angles)
    angles_diff = filt_angles-angles
    indices = [] # indices holds the slice range of interest
    for i in range(len(angles)):
        if abs(angles_diff[i]) < 0.05 and 1.45<angles[i]<1.75:
            indices.append(i)
    from itertools import groupby
    z = zip(indices, indices[1:])
    tmp = ([list(j) for i, j in groupby(z, key=lambda x: (x[1] - x[0]) <= 1)])
    tmp = sorted(tmp, key=len) # longest lists at the end
    if len(tmp)>1:
        maxtmp = np.array(max(tmp, key=len))
        slices = list(range(maxtmp[0][0], maxtmp[-1][-1]+1))
        slice_angles = [angles[i] for i in slices]
    else:
        slices = []
        slice_angles = [np.pi/2]
#    if not 0<abs(np.mean(slice_angles))<20/360*2*np.pi or not np.pi/2-20/360*2*np.pi<abs(np.mean(slice_angles))<20/360*2*np.pi+np.pi/2:
#        slice_angles = ['use eyes']
#        slices = ['use eyes']
    return slices, slice_angles

def findPlaneFromEllipses(bone, c1, c2, slices, head_x, head_y, head_angles):

    import findeyes as fer
    bone[abs(bone)<0.01]    = 0
    bone[~np.isfinite(bone)]= 0

    c = (c1+c2)/2.0
    c = np.roll(c,1)
    if slices != []:

        print '\n','Using ellipses to find midplane','\n'

        x=[]
        y=[]
        remaining_z = range(bone.shape[2]-1, slices[-1],-1)
        for i in remaining_z:
            x.append(np.mean(np.nonzero(bone[:,:,i])[1]))
            y.append(np.mean(np.nonzero(bone[:,:,i])[0]))
        centroids_array = np.concatenate((np.array([head_x,head_y,slices]).T, np.array([x,y,remaining_z]).T),axis=0).T
        centroids_array[~np.isfinite(centroids_array)]=0
        centroids_array[abs(centroids_array)<0.01] = 0
        centroids_array = rejectOutliers(centroids_array)
        from scipy.optimize import curve_fit
        def f(x,A,B):
            return A*x + B
        A_x1,B_x1 = curve_fit(f, centroids_array[2], centroids_array[0])[0] # your data x, y to fit
        A_y1,B_y1 = curve_fit(f, centroids_array[2], centroids_array[1])[0] # your data x, y to fit

        z = np.median(slices)
        x = A_x1*z + B_x1
        y = A_y1*z + B_y1
        # point that the plane goes through, p
        p = np.array([x, y, z]) # coordinates when z = mean(indices)
        print p
        v1 = np.array([A_x1/2, A_y1/2, 1])
        mean_angle = np.mean(head_angles)

        v2a = np.array([np.cos(mean_angle+np.pi/2), np.sin(mean_angle+np.pi/2), 0])
        v2b = (p - c)
        v2b = v2b/(np.sum(np.power(v2b,2)))
        v2 = (v2a+v2b)/2.0
        normal = np.cross(v1,v2)
        # a plane is ax+by+cz = d - find d
        d = np.dot(p,normal)
        # superpose plane onto slices
        a = normal[0]
        b = normal[1]
        c = normal[2]
        # plane equation is ax+by+cz = d
        # for each slice substitute slice number into z to get equation
    else:

        print '\n','Using eyes to find midplane, as ellipses were unreliable','\n'

        v1 = np.array([0,0,1])
        v2 = np.array([1,0,0])
        normal = np.cross(v1,v2)
        d = np.dot(c,normal)
        a = normal[0]
        b = normal[1]
        c = normal[2]
    return a,b,c,d,normal

def visualiseSingle(rotatedBone, a,b,c,d, slice_no):
    import plotly as py
    import plotly.graph_objs as go
    py.offline.init_notebook_mode()
    slice_ = rotatedBone[:,:,slice_no]
    heatmap = go.Heatmap(
            z = slice_,
            colorscale=[[0.0, 'rgb(160,160,160)'], 
                        [0.1111111111111111, 'rgb(20,20,20)'], 
                        [0.2222222222222222, 'rgb(40,40,40)'], 
                        [0.3333333333333333, 'rgb(100,100,100)'], 
                        [0.4444444444444444, 'rgb(120,120,120)'], 
                        [0.5555555555555556, 'rgb(140,140,140)'], 
                        [0.6666666666666666, 'rgb(160,160,160)'], 
                        [0.7777777777777778, 'rgb(170,170,170)'], 
                        [0.8888888888888888, 'rgb(250,250,250)'], 
                        [1.0, 'rgb(250,250,250)']]
            )
    x = np.linspace(0,slice_.shape[0], 2)
    y = (d-c*slice_no-a*x)/b
    midline = go.Scatter(
        x = x,
        y = y,
        mode = 'lines'
    )
    layout = go.Layout(
        width = 600,
        height= 600,
    #     title='Fig. 10. Slice number %i' % slice_no,
        title='Fig. 10. Slice of the midplane found', # title for report
        showlegend = False,
            margin= go.Margin(
            l=50,
            r=50,
            b=100,
            t=50,
            pad=4
            ),
#        yaxis=dict(
#            range=[50, 220]
#        ),
#        xaxis=dict(
#            range=[0, 220]
#        )
    )

    data = [heatmap, midline]
    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)
    return None

def correctPlaneParams(angle1, angle2, n, c1, c2):
    def rotationMatrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis/math.sqrt(np.dot(axis, axis))
        a = math.cos(theta/2.0)
        b, c, d = -axis*math.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    
    theta1 = -angle1/360.*2*np.pi
    theta2 = -angle2/360.*2*np.pi
    
    if abs(angle1) < 45:
        axis = [0,0,1]
        n1 = np.dot(rotationMatrix(axis,theta2*(1.-np.sin(theta1))), n)
        axis = [0,-1,0]
        normal = np.dot(rotationMatrix(axis,theta1), n1)

    elif abs(angle1) > 45:
        axis = [0,0,1]
        n1 = np.dot(rotationMatrix(axis,theta2*(np.sin(theta1))), n)
        axis = [0,-1,0]
        normal = np.dot(rotationMatrix(axis,np.pi/2.-theta1), n1)

    c = 0.5*(c1+c2)
    c = np.roll(c,1)
    d = np.dot(c,normal)
    a = normal[0]
    b = normal[1]
    c = normal[2]
    return a,b,c,d
    
    
    