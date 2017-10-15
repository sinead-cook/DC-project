import numpy as np
import nibabel as nib
import core
import SimpleITK as sitk
import os
import ui_log
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import findeyes

from skimage import filters
from scipy.ndimage.morphology import binary_erosion as be
from scipy.ndimage.morphology import binary_fill_holes as bfh
from scipy.ndimage.morphology import binary_opening as bo
from scipy.ndimage.morphology import binary_closing as bc
from scipy.ndimage.morphology import grey_dilation as gd
from scipy.ndimage.morphology import binary_dilation as bd

from skimage.segmentation import find_boundaries as boundaries
import skimage.measure
import skimage
import time

class master:

    # this class contains the "brains" of the software: it uses some functions that 
    # have been written in core and findeyes, but essentially is where all the analysis
    # is written

    def setup(self):
        import scandata

        self.scan = scandata.Scan()
        self.mask = scandata.Mask()
        self.methods = scandata.Methods()
    
        self.p_vol = np.nan
        self.b_vol = np.nan
        self.h_vol = np.nan
        self.v_vol = np.nan
        
    def saveMask(self, maskArray, saveName):
        import os
        scanName = os.path.splitext(os.path.basename(self.scan.path))[0]
        if self.scan.path.endswith('nii.gz'):
            scanName = os.path.splitext(scanName)[0]
        maskArraySwapped = np.swapaxes(maskArray, 0, 2)
        if self.scan.path.endswith('.nii') or self.scan.path.endswith('.nii.gz'):
            saveName = scanName + '-' + saveName
            img = nib.Nifti1Image(maskArray, self.scan.image.affine, self.scan.image.header)
            pathHead = os.path.split(self.scan.path)[0]
            savePath = os.path.join(pathHead, saveName)

            nib.save(img, savePath)
        else:
            import SimpleITK as sitk
            img = sitk.GetImageFromArray(maskArraySwapped, isVector = False)
            img.CopyInformation(self.scan.image) 
            if self.scan.path.endswith('nrrd'):
                saveName = scanName + '-' + saveName
                pathHead = os.path.split(self.scan.path)[0]
            elif self.scan.path.endswith('.dcm'):
                pathHead = os.path.split(os.path.split(self.scan.path)[0])[0]
            savePath = os.path.join(pathHead, saveName)
            sitk.WriteImage(img, savePath)

    def symmetryOutputs(self, tissueMask, saveName=None):

        left_mask = np.multiply(tissueMask, self.midplane_split==0)
        right_mask = np.multiply(tissueMask, self.midplane_split==1)

        if self.methods.lrTissueMasks == True:
            self.saveMask(left_mask, 'left_'+saveName)
            self.saveMask(right_mask, 'right_'+saveName)

        if self.methods.totalSymmVolumes == True:
            left_volume = (left_mask[left_mask!=0]).size*np.prod(self.scan.pixelspacing)
            right_volume = (right_mask[right_mask!=0]).size*np.prod(self.scan.pixelspacing)

        return left_volume, right_volume

    def extract(self, thresholded):
        import subprocess
        import os

        img = nib.Nifti1Image(thresholded, np.eye(4))

        if self.scan.path.endswith('.nii') or self.scan.path.endswith('.nii.gz'):
            pathHead = os.path.split(self.scan.path)[0]
        elif self.scan.path.endswith('nrrd'):
            pathHead = os.path.split(self.scan.path)[0]
        elif self.scan.path.endswith('.dcm'):
            pathHead = os.path.split(os.path.split(self.scan.path)[0])[0]

        nib.save(img, os.path.join(pathHead,'temp.nii.gz'))
        
        outputFileName = 'betlog.txt'
        outputFile = open(outputFileName, "w")
        p = subprocess.Popen(['which', 'bet'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
        result = p.stdout.read()
        print os.path.split(result)[0]
        os.environ['PATH'] += os.path.split(result)[0]
        proc = subprocess.Popen(['bet {} {} -f 0.3'.format(os.path.join(pathHead,'temp.nii.gz'), os.path.join(pathHead,'out.nii.gz'))], cwd=os.path.split(result)[0], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
        proc.stdin.close()
        proc.wait()
        result = proc.returncode
        outputFile.write(proc.stdout.read())

        img = nib.load(os.path.join(pathHead,'out.nii.gz'))
        data = img.get_data()

        os.remove(os.path.join(pathHead,'out.nii.gz'))
        os.remove(os.path.join(pathHead,'temp.nii.gz'))
        os.remove('betlog.txt')
        return data

    def readInScan(self):
        import psutil
        global path

        if self.scan.path.endswith('.nii') or self.scan.path.endswith('.nii.gz'):
            self.scan.image        = nib.load(self.scan.path)
            self.scan.array        = self.scan.image.get_data()
            self.scan.pixelspacing = self.scan.image.header['pixdim'][1:4]

        elif self.scan.path.endswith('nrrd'):
            self.scan.image        = sitk.ReadImage(self.scan.path)
            self.scan.array        = np.swapaxes(sitk.GetArrayFromImage(self.scan.image), 0, 2)
            self.scan.pixelspacing = self.scan.image.GetSpacing()

        elif self.scan.path.endswith('.dcm'):
            pathHead               = os.path.split(self.scan.path)[0]
            reader                 = sitk.ImageSeriesReader()
            dicomNames             = reader.GetGDCMSeriesFileNames(pathHead)
            reader.SetFileNames(dicomNames)
            self.scan.image        = reader.Execute()
            self.scan.array        = np.swapaxes(sitk.GetArrayFromImage(self.scan.image), 0, 2)
            self.scan.pixelspacing = self.scan.image.GetSpacing()

        self.scan.bone         = core.thresholdnp(self.scan.array, 1100, 1500)
        self.scan.softtissue   = core.thresholdnp(self.scan.array, 0, 80)

    def readIn(self, path):
        self.scan.path = str(path)
        self.readInScan()

        print ('Finished reading in scan. Start reshaping...')

    def reshapingScan(self):
        # reshape the numpy arrays so that they are isotropic (for circle detection)
        self.bone       = core.reshape(self.scan.bone, self.scan.pixelspacing)
        print 'reshaped bone shape is ', self.bone.shape
        del self.scan.bone
        self.softtissue = core.reshape(self.scan.softtissue, self.scan.pixelspacing)
        del self.scan.softtissue

        stringToPrint = 'Reshaping Complete. Starting midplane finder, finding eyes...'
        print ('Reshaping Complete. Starting midplane finder. Finding eyes...')

    def findingEyes(self):
        # find the eyes.
        # histogram data of the number of circles detected in the scans in a given geometric space (the scan is split into 100 cubes). The histogram data covers circles in 3 directions (the frequencies are summed together).
        H, edges, histData2   = findeyes.hist3dAll(self.softtissue)
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
        if xcoord>=self.bone.shape[0]/2:
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
        print 'Eyes found. Finding skew of the scan...'

        self.c1Reshaped, self.c2Reshaped = c1Reshaped, c2Reshaped
        self.c1, self.c2 = c1, c2 

    def findingSkew(self):

        c1Reshaped, c2Reshaped = self.c1Reshaped, self.c2Reshaped

        ccReshaped = 0.5*(c1Reshaped+c2Reshaped)
        angle1, angle2 = findeyes.anglesFromEyes(c1Reshaped, c2Reshaped, self.bone.shape)
        self.scan.eyes = np.divide(c1Reshaped, self.scan.pixelspacing), np.divide(c2Reshaped, self.scan.pixelspacing)
        
        while angle1<=-45: #skews are never that big
            angle1 = angle1+90

        while angle1>=45: #skews are never that big
            angle1 = angle1-90

        self.angle1 = angle1
        print 'Skew found. Rotating scan to correct skew...'

    def correctSkew(self):
        from scipy.ndimage.interpolation import rotate

        self.rotatedSofttissue = rotate(self.softtissue, -self.angle1, mode='nearest', axes=(0,2)) # want angle1 to be 90
        del self.softtissue

        self.rotatedBone = rotate(self.bone, -self.angle1, mode='nearest', axes=(0,2)) # want angle1 to be 90
        print 'Scan rotated. Finding eyes again in rotated scan...'

    def findingEyes2(self):
        import findeyes
        # Find Eyes Again
        H, edges, histData2 = findeyes.hist3dAll(self.rotatedSofttissue)
        firstEyeRange, secondEyeRange, certainty  = findeyes.ranges(H,edges)
        self.c1ReshapedRotated, self.c2ReshapedRotated = findeyes.coords(histData2, firstEyeRange, secondEyeRange)        

        stringToPrint = 'Eyes found in rotated scan. Finding ellipses in rotated scan...'
        print stringToPrint

    def ellipseFitting(self):

        import psutil
        # start = time.time()
        # while time.time() - start < 30:
        del self.rotatedSofttissue

        print psutil.virtual_memory()

        angs, xcentroids, ycentroids = core.ellipses(self.rotatedBone)

        # 'unreshaping' the results, so that the midplane will fit the original array
        # the slices of interest and their corresponding angles 
        slices, sliceAngles = core.selectEllipsesRange(angs)
        # the coordinates of the centroids of the ellipses in the slices of interest
        headx = [xcentroids[i] for i in slices] 
        heady = [ycentroids[i] for i in slices] 
        stringToPrint = 'Ellipses found in rotated scan, finding midplane normal in rotated scan...'
        print stringToPrint
        # find the normal of the midplane in the rotated system
        a,b,c,d,self.reshapedNormal = core.findPlaneFromEllipses(self.rotatedBone, self.c1ReshapedRotated, self.c2ReshapedRotated, slices, headx, heady, sliceAngles)
        print 'reshaped normal is ',self.reshapedNormal
        del self.rotatedBone

    def findingMidplane(self):

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
        
        print 'Transform normal in rotated scan to original coordinates ...'
        c1ReshapedRotated, c2ReshapedRotated = self.c1ReshapedRotated, self.c2ReshapedRotated
        c1Reshaped, c2Reshaped = self.c1Reshaped, self.c2Reshaped
        c1, c2 = self.c1, self.c2
        # swap c1ReshapedRotated, c2ReshapedRotated if necessary
        if np.linalg.norm(self.c1ReshapedRotated-self.c1Reshaped)>np.linalg.norm(self.c2ReshapedRotated-self.c2Reshaped):
            temp = c1ReshapedRotated
            c1ReshapedRotated = c2ReshapedRotated
            c2ReshapedRotated = temp
        else:
            pass
        
        crr = (c1ReshapedRotated-c2ReshapedRotated)
        cr  = (c1Reshaped-c2Reshaped)
        
        c = np.divide(cr, self.scan.pixelspacing)

        from scipy.optimize import minimize


        x = minimize(f,([1,0,0,0,1,0,0,0,1]), args=([[crr], [c]]))

        a,b,c,d,e,f,h,i,j = x.x

        T = np.asarray([[a, b, c],
                        [d, e, f],
                        [h, i, j]])

        normal = np.dot(T, self.reshapedNormal)
        normal = np.divide(normal, np.linalg.norm(normal))

        a,b,c = normal
        print 'normal is ', normal
        coord = 0.5*(c1+c2)
        d = np.dot(normal, coord)
        self.scan.params = a,b,c,d
        print 'a, b, c, d are ',self.scan.params

        del self.c1Reshaped, self.c2Reshaped, self.c1, self.c2, self.c1ReshapedRotated, self.c2ReshapedRotated

    def savingMasks(self):
        a,b,c,d= self.scan.params
        normal = np.array([a,b,c])

        print 'Midplane found for original coordinates.'

        if self.methods.midplaneMask==True:
            print 'Saving midplane mask...'

            crossShape = self.scan.array[:,:,0].shape
            self.mask.midplane = np.zeros(self.scan.array.shape)

            if abs(normal[1])>abs(normal[0]):
                for i in range(self.scan.array.shape[2]):
                    z = i
                    mask1 = np.fromfunction(lambda x,y: y > (d-c*i-a*x)/b-2, crossShape)
                    mask2 = np.fromfunction(lambda x,y: y < (d-c*i-a*x)/b+2, crossShape)
                    maski = np.multiply(mask1, mask2)
                    self.mask.midplane[:,:,i] = maski

            if abs(normal[1])<abs(normal[0]):
                for i in range(self.scan.array.shape[2]):
                    z = i
                    mask1 = np.fromfunction(lambda x,y: x > ((d-z*c-y*b)/a-2), crossShape)
                    mask2 = np.fromfunction(lambda x,y: x < ((d-z*c-y*b)/a+2), crossShape)
                    maski = np.multiply(mask1, mask2)
                    self.mask.midplane[:,:,i] = maski

            print 'Midplane mask created.','\n'

            self.saveMask(self.mask.midplane, 'midplane.nii.gz')

    def volumeAnalysis(self):
        print 'running vol analysis'
        import scipy.ndimage as ndi

        if self.methods.segmentation == True:
            self.readInScan()
        skin_mask = ndi.filters.gaussian_filter(self.scan.array, 1)
        skin_mask = np.multiply(skin_mask, (self.scan.array>-200).astype(int))
        array1 = np.multiply(skin_mask, (skin_mask>0.0).astype(int))
        self.array1 = np.multiply(array1, (array1>-200).astype(int))

    def skinOrbital(self):
        # skull
        skull_mask = self.array1>84.0
        array1 = self.array1
        array2 = np.multiply(array1, (array1<84.0).astype(int))
        # orbital
        orbital_mask = skull_mask

        for i in range(5):
            orbital_mask = bd(skull_mask)

        array3 = np.multiply(array2, (orbital_mask==False).astype(int))

        # cerebral ventricle
        self.array4 = np.multiply(array3, (array3>0.0).astype(int))

    def brainExtraction(self):

        # BET FSL
        array5 = self.extract(self.array4)

        del self.array4
        self.brain_mask = np.multiply(bc(bfh(bd(array5))).astype(int), array5)
        self.b_vol = (self.brain_mask[self.brain_mask!=0]).size*np.prod(self.scan.pixelspacing)

        print 'FSL BET tool complete'

        if self.methods.wholeBrain==True and self.methods.tissueMasks==True: 
            self.saveMask(self.brain_mask, 'wholeBrain_mask.nii.gz')

        self.array5 = array5
    def findingVentricles(self):

        if self.methods.ventricles == True: # only find ventricles if user is interested in them
            print 'finding ventricles'

        # Ventricle Analysis
            brainArea = []
            for i in range(self.brain_mask.shape[2]):
                brainArea.append(np.count_nonzero(self.brain_mask[:,:,i]))
            midsliceIndex = np.argmax(brainArea)
            brain_mask=self.brain_mask
            # get ventricles
            ventr_mask1 = np.multiply((brain_mask>0).astype(int), (brain_mask<22).astype(int))
            ventr_mask2 = bd(ventr_mask1)
            ventr_mask3 = be(ventr_mask2)
            ventr_mask4 = bd(ventr_mask3)
            ventr_loop = ventr_mask3
            
            for ind in range(1):
                labels = skimage.measure.label(ventr_loop, connectivity=1)
                props = skimage.measure.regionprops(labels)
                v = [p.area for p in props]
                x = 1
                ind = v.index(max(v))
                vcoords = props[ind].coords
                while abs(np.mean(vcoords.transpose()[2])-midsliceIndex)>self.scan.array.shape[2]/10:
                    ind = np.argsort(v)[-x]
                    vcoords = props[ind].coords
                    x = x+1
                ventr_loop = np.zeros((self.scan.array.shape))
                for i in range(len(vcoords)):
                    a,b,c = vcoords[i]
                    ventr_loop[a,b,c]=1
                ventr_loop = bd(be(ventr_loop))


            for i in range(3):
                labels = skimage.measure.label(ventr_loop, connectivity=1)
                props = skimage.measure.regionprops(labels)
                v = [p.area for p in props]
                ind = v.index(max(v))
                vcoords = props[ind].coords
                ventr_loop = np.zeros((self.scan.array.shape))
                for i in range(len(vcoords)):
                    a,b,c = vcoords[i]
                    ventr_loop[a,b,c]=1
                ventr_loop = bd(be(ventr_loop)) 
                
            labels = skimage.measure.label(ventr_loop, connectivity=1)
            props = skimage.measure.regionprops(labels)
            v = [p.area for p in props]
            ind = v.index(max(v))
            vcoords = props[ind].coords
            ventr_loop = np.zeros((self.scan.array.shape))
            for i in range(len(vcoords)):
                a,b,c = vcoords[i]
                ventr_loop[a,b,c]=1
                
            ventr_loop = bd(be(ventr_loop))

            v_mask = bfh(ventr_loop)

            self.v_mask = (v_mask).astype(np.float64)

            print v_mask.shape
            print self.scan.array.shape

            if self.methods.tissueMasks==True:
                print 'saving ventricle mask'
                self.saveMask(self.v_mask, 'ventricles_mask.nii.gz')
                print 'ventricle mask saved'

            self.v_vol = (self.v_mask[self.v_mask!=0]).size*np.prod(self.scan.pixelspacing)


    def findingHaematoma(self):
        
        if self.methods.haematoma == True or self.methods.parenchyma_mask == True:
            # need to find haematoma to know the parenchyma mask or the haematoma mask
            # get haematoma
            print 'finding haematoma'
            h_mask = (self.array5>50.0).astype(float)*1000
            array6 = np.multiply(self.brain_mask, h_mask)
            array7 = be(array6)
            array8 = bc((array7))
            array9 = bfh((array8))
            array10 = bd(bd((array8)))
            array11 = bfh(bd(array10))

            h_loop = array11
                
            for i in range(4):
                labels = skimage.measure.label(h_loop, connectivity=1)
                props = skimage.measure.regionprops(labels)
                h = [p.area for p in props]
                ind = h.index(max(h))
                hcoords = props[ind].coords
                h_loop = np.zeros((self.scan.array.shape))
                for i in range(len(hcoords)):
                    a,b,c = hcoords[i]
                    h_loop[a,b,c]=1
                h_loop = be(h_loop)

            labels = skimage.measure.label(h_loop, connectivity=1)
            props = skimage.measure.regionprops(labels)
            h = [p.area for p in props]
            ind = h.index(max(h))
            hcoords = props[ind].coords
            h_loop = np.zeros((self.scan.array.shape))
            for i in range(len(hcoords)):
                a,b,c = hcoords[i]
                h_loop[a,b,c]=1
            h_loop = bd(h_loop)

            h_mask = bfh(h_loop)

            self.h_mask = (h_mask).astype(np.float64)

            if self.tissueMasks == True:
                self.saveMask(self.h_mask, 'haematoma_mask.nii.gz')

            self.h_vol = (self.h_mask[self.h_mask!=0]).size*np.prod(self.scan.pixelspacing)

    def savingExtraMasks(self):
        
        left_h_vol = np.nan
        right_h_vol = np.nan
        left_v_vol = np.nan
        right_v_vol = np.nan
        left_b_vol = np.nan
        right_b_vol = np.nan
        left_p_vol = np.nan
        right_p_vol = np.nan

        if self.methods.parenchyma == True:
            parenchyma_mask = np.multiply(self.brain_mask, (self.h_mask==0).astype(np.float64)) # no haematoma
            self.p_vol = (parenchyma_mask[parenchyma_mask!=0]).size*np.prod(self.scan.pixelspacing)
            if self.tissueMasks == True:
                self.saveMask(parenchyma_mask, 'parenchyma_mask.nii.gz')

        if self.methods.symmetry == True: # this means that either lrTissueMasks is checked or that totalSymmVolumes is checked
            # volumes analysis
            a,b,c,d = self.scan.params
            normal = [a,b,c]

            crossShape = self.scan.array[:,:,0].shape

            midplane = np.zeros(self.scan.array.shape)

            if abs(normal[1])>abs(normal[0]):
                for i in range(self.scan.array.shape[2]):
                    z = i
                    maski = np.fromfunction(lambda x,y: y > (d-c*i-a*x)/b, crossShape)
                    midplane[:,:,i] = maski

            if abs(normal[1])<abs(normal[0]):
                for i in range(self.scan.array.shape[2]):
                    z = i
                    maski = np.fromfunction(lambda x,y: x > ((d-i*c-y*b)/a), crossShape)
                    midplane[:,:,i] = maski

            self.midplane_split = np.flip(midplane,2)
            
            if self.methods.haematoma == True:
                left_h_vol, right_h_vol = self.symmetryOutputs(self.h_mask, 'haematoma_mask.nii.gz')
                print 'left haematoma volume is ', left_h_vol
                print 'right haematoma volume is ', right_h_vol

            if self.methods.ventricles == True:
                left_v_vol, right_v_vol = self.symmetryOutputs(self.v_mask, 'ventricles_mask.nii.gz')
                print 'left ventricles volume is ', left_v_vol
                print 'right ventricles volume is ', right_v_vol

            if self.methods.wholeBrain == True:
                left_b_vol, right_b_vol = self.symmetryOutputs(self.brain_mask, 'wholebrain_mask.nii.gz')
                print 'left brain volume is ', left_b_vol
                print 'right brain volume is ', right_b_vol

            if self.methods.parenchyma == True:
                left_p_vol, right_p_vol = self.symmetryOutputs(parenchyma_mask, 'parenchyma_mask.nii.gz')
                print 'left parenchyma volume is ', left_p_vol
                print 'right parenchyma volume is ', right_p_vol

        p_vol = self.p_vol
        b_vol = self.b_vol
        v_vol = self.v_vol
        h_vol = self.h_vol
        self.output_i = [self.scan.path, left_h_vol, right_h_vol, left_v_vol, right_v_vol, left_b_vol, right_b_vol, left_p_vol, right_p_vol, h_vol, v_vol, b_vol, p_vol]
           
