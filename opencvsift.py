#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:31:52 2016

@author: Sinead
"""
# http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html

# feature matching
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

# make sure working directory is set
import matplotlib
import numpy as np
import cv2
import script


PathDicom = '/Volumes/Backup Data/ASDH Samples/Sample1/Pre-operative/R-N11-109/HeadSpi  1.0  J40s  3'
ArrayDicom=script.dicom2np(PathDicom)
image_array=ArrayDicom[:,:,100]
matplotlib.image.imsave('name.png', image_array)

img = cv2.imread('name.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp,img)