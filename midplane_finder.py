import os
os.chdir('/Users/Sinead/DC-project/')
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import script
import plotly 
plotly.offline.init_notebook_mode()
import find_eyes_reshape as fer
import nibabel as nib
import vtk

def midplane_finder(PathDicom):

    # threshold CT scan for eyes and bone and store in 2 different arrays

    if PathDicom.endswith('.nii'):
        thresholded_bone=script.threshim_nii(700,1500)
        array_bone, ConstPixelSpacing = script.vtk2np_nii(thresholded_bone)
        thresholded_eyes = script.threshim_nii(0,80) # remove soft tissue
        array_eyes, ConstPixelSpacing = script.vtk2np_nii(thresholded_eyes)
        thresholded_visual=script.threshim_nii(-100,1500) 
        array_visual, ConstPixelSpacing = script.vtk2np_nii(thresholded_visual) 
    else:
        thresholded_bone=script.threshim_dicom(700,1500)
        array_bone, ConstPixelSpacing = script.vtk2np(thresholded_bone)
        thresholded_eyes = script.threshim_dicom(0,80) # remove soft tissue
        array_eyes, ConstPixelSpacing = script.vtk2np(thresholded_eyes)
        thresholded_visual=script.threshim_dicom(-100,1500) 
        array_visual, ConstPixelSpacing = script.vtk2np(thresholded_visual) 

    # reshape the arrays - don't run this twice
    array_bone = script.reshape(array_bone, ConstPixelSpacing, 220, 160, array_bone.shape[0], array_bone.shape[2])
    array_eyes = script.reshape(array_eyes, ConstPixelSpacing, 220, 160, array_eyes.shape[0], array_eyes.shape[2])
    array_visual = script.reshape(array_visual, ConstPixelSpacing, 220, 160, array_visual.shape[0], array_visual.shape[2])

    H, edges, data, hist_data_2c = fer.hist3d_all(array_eyes)

    ranges_1, ranges_2, certainty = fer.ranges(H,edges)
    c1,c2 = fer.coords(hist_data_2c, ranges_1, ranges_2)
    # fer.check_coords(c1,c2,array_eyes)

    angle1, angle2 = fer.angles_from_eyes(c1,c2)
    rotated1, rotated2 = fer.correct_skews(angle1,angle2,array_bone)
    _, array_visual = fer.correct_skews(angle1,angle2,array_visual)

    angles, xcentroids, ycentroids, xmajor_axis_points, ymajor_axis_points = script.ellipses(rotated2)

    slices, slice_angles = script.select_ellipses_range(angles)

    head_angles = [angles[i] for i in slices]
    head_x = [xcentroids[i] for i in slices] 
    head_y = [ycentroids[i] for i in slices] 

    a,b,c,d = script.find_plane_from_ellipses(rotated2, slices, head_x, head_y, head_angles)

    script.visualise_single(array_visual, a,b,c,d, slice_no)

    script.save_midplane(array_visual, a,b,c,d)
    
    return None