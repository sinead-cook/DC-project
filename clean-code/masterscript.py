#!/usr/bin/env python
# Copyright (c) Sinead Cook 2017. All rights reserved.
# This program or module is free software: you can redistribute it and/or
# modify it.

import scandata
import core
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from Tkinter import Tk
from tkFileDialog import askopenfilename


class master:
    def run(self):

        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        self.path = askopenfilename()

        self.scan = scandata.Scan()
        self.analysis()


    def analysis(self):
    
        self.midplaneFinder()
        self.midplaneMask()


    def midplaneFinder(self):
        import findeyes 

        print '\n','\n','Reading in and thresholding scan...','\n','\n'


        if self.path.endswith('.nii') or self.path.endswith('.nii.gz'):
            array, self.scan.pixelspacing = core.nifti2np(self.path)
            bone       = core.thresholdnp(array, 1100, 1500)
            softtissue = core.thresholdnp(array, 0, 80)
            # visual     = core.thresholdnp(self.scan.array, -100, 1500)

        elif self.path.endswith('.dcm'):
            self.scan.array, self.scan.pixelspacing = core.dicom2np(self.path)
            bone       = core.thresholdnp(array, 1100, 1500)
            softtissue = core.thresholdnp(array, 0, 80)
            # visual     = core.thresholdnp(self.scan.array, -100, 1500)

        print '\n','\n',"Finished reading in scan",'\n','\n'

        bone       = core.reshape(bone,       self.scan.pixelspacing, bone.shape[0],       bone.shape[2])
        softtissue = core.reshape(softtissue, self.scan.pixelspacing, softtissue.shape[0], softtissue.shape[2])
        array      = core.reshape(array, self.scan.pixelspacing, softtissue.shape[0], softtissue.shape[2])

        print '\n','\n','Reshaping Complete','\n','\n'

        # plt.imshow(self.scan.softtissue[:,:,50])
        # plt.show()

        H, edges, data, hist_data_2c   = findeyes.hist3d_all(softtissue)
        ranges_1, ranges_2, certainty  = findeyes.ranges(H,edges)
        c1,c2 = findeyes.coords(hist_data_2c, ranges_1, ranges_2)

        angle1, angle2               = findeyes.angles_from_eyes(c1,c2)
        _, self.scan.bone            = findeyes.correct_skews(angle1,angle2,bone)
        _, self.scan.softtissue      = findeyes.correct_skews(angle1,angle2,softtissue)
        _, self.scan.array           = findeyes.correct_skews(angle1,angle2,array)

        angles, xcentroids, ycentroids, xmajor_axis_points, ymajor_axis_points = core.ellipses(self.scan.bone)
        slices, slice_angles = core.selectellipsesrange(angles)

        head_x = [xcentroids[i] for i in slices] 
        head_y = [ycentroids[i] for i in slices] 

        a,b,c,d = script.find_plane_from_ellipses(array_bone2,array_eyes2, slices, head_x, head_y, slice_angles)

        self.scan.params = a,b,c,d

    def midplaneMask(self):
        import nibabel as nib
        import os

        a,b,c,d = planes[s]
        shape = (self.scan.array).shape
        mask = -1000*np.ones((shape))
        x = np.arange(0,shape[0],1)
        z = np.linspace(0,shape[2],shape[0])

        xx, yy = np.meshgrid(z, (d-c*z-a*x)/b)
        y = yy.astype(int)

        for k in range(shape[1]):
            for i in range(shape[2]):
                mask[y[k][i]-1:y[k][i]+1, k, i] = 1000

        img = nib.Nifti1Image(mask, np.eye(4))
        midplaneMaskPath = os.path.join(os.path.split(path)[0], 'midplane.nii')
        nib.save(img, midplaneMaskPath)                                                        

if __name__ == "__main__":
    master.run(master())
