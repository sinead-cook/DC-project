import numpy as np
import script
import matplotlib.pyplot as plt
import cv2
import cv2.cv as cv
import matplotlib.image

def single_slice(axis_no, thresholded_np, slice_no):
    if axis_no == 0:
        matplotlib.image.imsave('img.png', thresholded_np[slice_no,:,:]) 
    elif axis_no == 1:
        matplotlib.image.imsave('img.png', thresholded_np[:,slice_no,:]) 
    elif axis_no == 2:
        matplotlib.image.imsave('img.png', thresholded_np[:,:,slice_no]) 
    else:
        print 'axis_no must be 0, 1 or 2'
        return None
    img = cv2.imread('img.png')
    img = cv2.resize(img, None, fx=2, fy=2)
    img = cv2.medianBlur(img,3)
    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,1,50, 
                               param1=50,param2=30,minRadius=5,maxRadius=30)
    try:
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    except:
        pass
    plt.clf()
    plt.imshow(cimg)
    plt.axis('equal')
    plt.grid('on')
    return plt.show()

def all_slices(axis_no, thresholded_np, circles_per_slice):
    import matplotlib
    import cv2
    import cv2.cv as cv
    circles_data = np.zeros((thresholded_np.shape[axis_no], circles_per_slice+1, 3))
    for i in range(thresholded_np.shape[axis_no]):
        # to change dimension, change where i is 
        if axis_no == 0:
            matplotlib.image.imsave('img.png', thresholded_np[i,:,:]) 
            img = cv2.imread('img.png')
        elif axis_no == 1:
            matplotlib.image.imsave('img.png', thresholded_np[:,i,:]) 
            img = cv2.imread('img.png')
        elif axis_no == 2:
            matplotlib.image.imsave('img.png', thresholded_np[:,:,i]) 
            img = cv2.imread('img.png')
        else:
            print 'axis_no must be 0, 1 or 2'
            return None            
        img = cv2.resize(img, None, fx=2, fy=2)
        img = cv2.medianBlur(img,3)
        cimg= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,1,50, 
                               param1=50,param2=30,minRadius=5,maxRadius=30)
        try:
            circles_data[i,:,:] = circles[0][0:circles_per_slice+1]
        except:
            pass
    return circles_data

def indexed_data(circles_data, circles_per_slice):
    # hist_data is the same size as circles_data but has an additional column (for the explicit 
    # slice number). hist_data is x,y,z,r
    hist_data = np.zeros((circles_data.shape[0], circles_data.shape[1], circles_data.shape[2]+1))
 
    for i in range(hist_data.shape[0]):
        
        hist_data[i,0:circles_per_slice+1, 0:2] = circles_data[i,0:circles_per_slice+1, 0:2] 
        #first 2 cols of every slice in circles data assigned to first 2 cols of every slice in hist_data
        
        hist_data[i,0:circles_per_slice+1, 3] = circles_data[i,0:circles_per_slice+1, 2] 
        # 3rd col of every slice in circles data assigned to 4rd col of every slice in hist_data (radii data)
        
        hist_data[i,0:circles_per_slice+1,2]= i*2 # fill in index and stretch by factor of 2
    return hist_data

def reshape(hist_data0, hist_data1, hist_data2):
           # if axis_no is 0: 1st column is z, 2nd column is y, 3rd column is x. 
        # if axis_no is 1: 1st column is z, 2nd column is x, 3rd column is y.
        # if axis_no is 2: 1st column is y, 2nd column is x, 3rd column is z.
    x0 = hist_data0[:,:,2].ravel()
    y0 = hist_data0[:,:,1].ravel()
    z0 = hist_data0[:,:,0].ravel()
    r0 = hist_data0[:,:,3].ravel()

    x1 = hist_data1[:,:,1].ravel()
    y1 = hist_data1[:,:,2].ravel()
    z1 = hist_data1[:,:,0].ravel()
    r1 = hist_data1[:,:,3].ravel()

    x2 = hist_data2[:,:,1].ravel()
    y2 = hist_data2[:,:,0].ravel()
    z2 = hist_data2[:,:,2].ravel()
    r2 = hist_data2[:,:,3].ravel()

    x = np.append(x0,np.append(x1,x2))
    y = np.append(y0,np.append(y1,y2))
    z = np.append(z0,np.append(z1,z2))
    r = np.append(r0,np.append(r1,r2))

    hist_data = np.array([x,y,z,r]).T
    return hist_data

def hist3d(hist_data):
    H, edges = np.histogramdd(hist_data[:, 0:3]) 

    # remove all the data points on the axes 
    H[0,:,:] = 0
    H[:,0,:] = 0
    H[:,:,0] = 0

    flat_H = H.flatten()

    mid_edges_x = np.zeros((len(edges[0])-1))
    mid_edges_y = np.zeros((len(edges[1])-1))
    mid_edges_z = np.zeros((len(edges[2])-1))

    for i in range(len(mid_edges_x)):
        mid_edges_x[i] = (edges[0][i]+edges[0][i+1])/2
    for i in range(len(mid_edges_y)):
        mid_edges_y[i] = (edges[1][i]+edges[1][i+1])/2
    for i in range(len(mid_edges_z)):
        mid_edges_z[i] = (edges[2][i]+edges[2][i+1])/2

    z = np.tile(mid_edges_z, len(mid_edges_y)*len(mid_edges_x))
    y = np.tile(np.repeat(mid_edges_y, len(mid_edges_z)), len(mid_edges_x))
    x = np.repeat(mid_edges_x, len(mid_edges_x)*len(mid_edges_z))

    data = np.array([x,y,z,flat_H])

    for i in range(data.shape[0]):
        data = np.array(data[:,data[i]!=0])
    return data, H, edges