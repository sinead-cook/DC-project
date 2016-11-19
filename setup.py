#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:13:34 2016

@author: Sinead
"""

import sys
import os
import matplotlib.pyplot as pyplot
import numpy as np
import vtk
import plotly
import enum
import IPython
from IPython.display import Image
import dicom #pydicom is dicom!!!
import numpy

os.chdir('/Users/Sinead/DC-project')
from ds_from_series import ds_from_series
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from functions import vtk_show, vtkImageToNumPy, plotHeatmap