#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:36:37 2016

@author: Sinead
"""
# get all the file directories and modules needed


import os
import sys
import numpy as np
import nibabel as nib
import pydicom as dicom
import matplotlib.pyplot as pyplot


# add sample 1 to path
sys.path.append('/Volumes/Backup Data/ASDH Samples/Sample1/Pre-operative/R-N11-109/HeadSpi  1.0  J40s  3')
sys.path.append('/Volumes/Backup Data/ASDH Samples/Sample1/Post-operative/R-N11-109/HeadSpi  1.0  J40s  3')

# add sample 2 to path
sys.path.append('/Volumes/Backup Data/ASDH Samples/Sample2/Original')
