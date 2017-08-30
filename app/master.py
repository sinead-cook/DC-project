import os
os.chdir('/Volumes/SINEADUSB/DC-project/app')
import matplotlib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tkinter import Tk
from tkFileDialog import askopenfilename
import numpy as np
import findeyes
import nibabel as nib
import scandata
import core
import SimpleITK as sitk
import masterscript
import ui_startupdlg
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import scipy
from scipy.ndimage.interpolation import rotate

import numpy as np
import core
import pylab as plt
import nibabel as nib
import os
from skimage import filters
import scipy.ndimage as ndi
from scipy.ndimage.morphology import binary_erosion as be
from scipy.ndimage.morphology import binary_fill_holes as bfh
from scipy.ndimage.morphology import binary_dilation as bd
from scipy.ndimage.morphology import binary_opening as bo
from scipy.ndimage.morphology import binary_closing as bc
from scipy.ndimage.morphology import grey_dilation as gd
import time
from skimage.segmentation import find_boundaries as boundaries
import skimage
import SimpleITK as sitk

class master:
    def setup(self):
        self.scan = scandata.Scan()
        self.mask = scandata.Mask()
    
    def selectScan(self):
        root = Tk()
        # ask for scan file
        # we don't want a full GUI, so keep the root window from appearing
#         root.withdraw() 
        root.update()
        self.scan.path = askopenfilename()
        root.quit()
        root.destroy()
        
    def analyse(self):
        if self.scan.path.endswith('.nii') or self.scan.path.endswith('.nii.gz') or self.scan.path.endswith('nrrd'):
            self.scan.image        = sitk.ReadImage(self.scan.path)
            self.scan.array        = np.swapaxes(sitk.GetArrayFromImage(self.scan.image), 0, 2)
            self.scan.pixelspacing = self.scan.image.GetSpacing()
            self.scan.bone         = core.thresholdnp(self.scan.array, 1100, 1500)
            self.scan.softtissue   = core.thresholdnp(self.scan.array, 0, 80)

        elif self.scan.path.endswith('.dcm'):
            pathHead               = os.path.split(self.scan.path)[0]
            reader                 = sitk.ImageSeriesReader()
            dicomNames             = reader.GetGDCMSeriesFileNames(pathHead)
            reader.SetFileNames(dicomNames)
            self.scan.image        = reader.Execute()
            self.scan.array        = np.swapaxes(sitk.GetArrayFromImage(self.scan.image), 0, 2)
            self.scan.pixelspacing = self.scan.image.GetSpacing()
            self.scan.bone         = core.thresholdnp(self.scan.array, 900, 1500)
            self.scan.softtissue   = core.thresholdnp(self.scan.array, 0, 80)

        print 'Finished reading in scan. Starting reshaping...','\n'

        # reshape the numpy arrays so that they are isotropic (for circle detection)
        bone       = core.reshape(self.scan.bone, self.scan.pixelspacing)
        softtissue = core.reshape(self.scan.softtissue, self.scan.pixelspacing)

        print 'Reshaping Complete. Starting midplane finder, finding eyes...','\n'

        # find the eyes.
        # histogram data of the number of circles detected in the scans in a given geometric space (the scan is split into 100 cubes). The histogram data covers circles in 3 directions (the frequencies are summed together).
        H, edges, histData2   = findeyes.hist3dAll(softtissue)
        # firstEyeRange, secondEyeRange contains the geometric space of the eyes (in the coordinate system we are currently using)
        firstEyeRange, secondEyeRange, certainty  = findeyes.ranges(H,edges)
        # we can now find the coordinates of the center of the eyes, using the histogram data
        c1Reshaped,c2Reshaped = findeyes.coords(histData2, firstEyeRange, secondEyeRange)
        # these coordinates (c1Reshaped etc.) are in the reshaped, isotropic coordinate system. The coordinates of the eyes in the original coordinate system of the CT scan are in c1 and c2.
        
        cc = 0.5*(c1Reshaped + c2Reshaped)
        xcoord = cc[1]
        ycoord = cc[0]
        c1, c2 = np.divide(c1Reshaped, self.scan.pixelspacing), np.divide(c2Reshaped, self.scan.pixelspacing)

        # adjusting so that c1 is always less than c2 in the x coordinate, if the eyes are in the upper half of the scan
        # adjusting so that c1 is always greater than c2 in the x coordinate, if the eyes are in the lower half of the scan

        # upper half 
        if xcoord>=bone.shape[0]/2:
            if c1[1]<c2[1]: # x coords of c1
                pass 
            else: 
                temp = c2Reshaped
                c2Reshaped = c1Reshaped
                c1Reshaped = temp
        # lower half 
        else:
            if c1[1]>c2[1]: # x coords of c1
                temp = c2Reshaped
                c2Reshaped = c1Reshaped
                c1Reshaped = temp 
            else: 
                pass
        print 'Eyes found. Finding skew of the scan...','\n'

        ccReshaped = 0.5*(c1Reshaped+c2Reshaped)
        angle1, angle2 = findeyes.anglesFromEyes(c1Reshaped, c2Reshaped, bone.shape)
        
        self.scan.eyes = np.divide(c1Reshaped, self.scan.pixelspacing), np.divide(c2Reshaped, self.scan.pixelspacing)
        
        while angle1<=-45: #skews are never that big
            angle1 = angle1+90

        while angle1>=45: #skews are never that big
            angle1 = angle1-90
        print 'Skew found. Rotating scan to correct skew...','\n'

        rotatedSofttissue = rotate(softtissue, -angle1, mode='nearest', axes=(0,2)) # want angle1 to be 90

        rotatedBone = rotate(bone, -angle1, mode='nearest', axes=(0,2)) # want angle1 to be 90
        # rotatedBone = rotate(rotatedBone1, rotAngle, mode='nearest', axes=(0,2))
        print 'Scan rotated. Finding eyes again in rotated scan...','\n'

        # Find Eyes Again
        H, edges, histData2 = findeyes.hist3dAll(rotatedSofttissue)
        firstEyeRange, secondEyeRange, certainty  = findeyes.ranges(H,edges)
        c1ReshapedRotated, c2ReshapedRotated = findeyes.coords(histData2, firstEyeRange, secondEyeRange)        

        print 'Eyes found in rotated scan. Finding ellipses in rotated scan...','\n'

        angs, xcentroids, ycentroids = core.ellipses(rotatedBone)
        # 'unreshaping' the results, so that the midplane will fit the original array
        # the slices of interest and their corresponding angles 
        slices, sliceAngles = core.selectEllipsesRange(angs)
        # the coordinates of the centroids of the ellipses in the slices of interest
        headx = [xcentroids[i] for i in slices] 
        heady = [ycentroids[i] for i in slices] 
        print 'Ellipses found in rotated scan, finding midplane normal in rotated scan...','\n'

        # find the normal of the midplane in the rotated system
        a,b,c,d,reshapedNormal = core.findPlaneFromEllipses(rotatedBone, c1ReshapedRotated, c2ReshapedRotated, slices, headx, heady, sliceAngles)

        def f(params, args):
            # optimisation function for finding the transformation between the original
            # and reshaped & rotated coordinate systems

            a,b,c,d,e,f,h,i,j = params
            vector =  args[0][0]
            comparison = args[1][0]

            T = np.asarray([[a, b, c],
                            [d, e, f],
                            [h, i, j]])

            v1 = np.dot(T, vector)

            v2 = comparison
            diff = np.linalg.norm(v1-v2)
            return diff
        
        print 'Transform normal in rotated scan to original coordinates ...','\n'

        # swap c1ReshapedRotated, c2ReshapedRotated if necessary
        if np.linalg.norm(c1ReshapedRotated-c1Reshaped)>np.linalg.norm(c2ReshapedRotated-c2Reshaped):
            temp = c1ReshapedRotated
            c1ReshapedRotated = c2ReshapedRotated
            c2ReshapedRotated = temp
        else:
            pass
        
        crr = (c1ReshapedRotated-c2ReshapedRotated)
        cr  = (c1Reshaped-c2Reshaped)
        
        c = np.divide(cr, self.scan.pixelspacing)

        x = scipy.optimize.minimize(f,([1,0,0,0,1,0,0,0,1]), args=([[crr], [c]]))

        a,b,c,d,e,f,h,i,j = x.x

        T = np.asarray([[a, b, c],
                        [d, e, f],
                        [h, i, j]])

        normal = np.dot(T, reshapedNormal)
        normal = np.divide(normal, np.linalg.norm(normal))

        a,b,c = normal
        coord = 0.5*(c1+c2)
        d = np.dot(normal, coord)
        self.scan.params = a,b,c,d
        print 'Midplane found for original coordinates. Saving midplane mask...','\n'

        crossShape = self.scan.bone[:,:,0].shape
        self.mask.midplane = np.zeros(self.scan.bone.shape)

        if abs(normal[1])>abs(normal[0]):
            print 0
            for i in range(self.scan.bone.shape[2]):
                z = i
                mask1 = np.fromfunction(lambda x,y: y > (d-c*i-a*x)/b-2, crossShape)
                mask2 = np.fromfunction(lambda x,y: y < (d-c*i-a*x)/b+2, crossShape)
                maski = np.multiply(mask1, mask2)
                self.mask.midplane[:,:,i] = maski

        if abs(normal[1])<abs(normal[0]):
            print 1
            for i in range(self.scan.bone.shape[2]):
                z = i
                mask1 = np.fromfunction(lambda x,y: x > ((d-z*c-y*b)/a-2), crossShape)
                mask2 = np.fromfunction(lambda x,y: x < ((d-z*c-y*b)/a+2), crossShape)
                maski = np.multiply(mask1, mask2)
                self.mask.midplane[:,:,i] = maski

        print 'Midplane mask created.','\n'

if __name__ == "__main__":
    m = master()
    m.setup()
    m.selectScan() # this will create the path to the scan
    m.analyse()

# m = master()