import os
os.chdir('/Users/Sinead/vtkpython/bin')
import subprocess
subprocess.call("./vtkpython")
import vtk
import numpy as np
import sys
sys.path.append('/anaconda/lib/python2.7/site-packages')
import IPython
os.chdir('/Users/Sinead/DC-project')
from functions import vtkImageToNumPy
from functions import plotHeatmap
os.chdir('/Users/Sinead/vtkpython/bin/vtk/util')
import numpy_support