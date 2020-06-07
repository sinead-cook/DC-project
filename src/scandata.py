#!/usr/bin/env python
# Copyright (c) 2007-8 Qtrac Ltd. All rights reserved.
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 2 of the License, or
# version 3 of the License, or (at your option) any later version. It is
# provided for educational purposes and is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.


from PyQt4.QtCore import *
from PyQt4.QtXml import *


CODEC = "UTF-8"
NEWPARA = unichr(0x2029)
NEWLINE = unichr(0x2028)
DATEFORMAT = "ddd MMM d, yyyy"


class Scan(object):
    """A Movie object holds the details of a movie.
    
    The data held are the title, year, minutes length, date acquired,
    and notes. If the year is unknown it is set to 1890. If the minutes
    is unknown it is set to 0. The title and notes are held as QStrings,
    and the notes may contain embedded newlines. Both are plain text,
    and can contain any Unicode characters. The title cannot contain
    newlines or tabs, but the notes can contain both. The date acquired
    is held as a QDate.
    """

    def __init__(self, array=None, spacing=None, params=None, notes=None, path=None, eyecoordinates=None):
        self.array    = array
        self.spacing  = spacing
        self.params   = params # set this to a, b, c, d
        self.notes    = notes
        self.path     = path
        self.eyes     = eyecoordinates # set this to c1, c2 (the coordinates of each eye)


    @staticmethod
    def formats():
        return "*.dcm *.nii *.nii.gz *.nrrd"

class Methods(object):
    def __init__(self, haematoma=None, parenchyma=None, segmentation=None, symmetry=None, ventricles=None, wholeBrain=None, lrTissueMasks=None, midplaneMask=None, tissueMasks=None, totalSymmVolumes=None, totalVolumes=None):
        self.haematoma = haematoma
        self.parenchyma = parenchyma
        self.segmentation = segmentation
        self.symmetry = symmetry
        self.ventricles = ventricles
        self.wholeBrain = wholeBrain
        self.lrTissueMasks = lrTissueMasks
        self.midplaneMask = midplaneMask
        self.tissueMasks = tissueMasks
        self.totalSymmVolumes = totalSymmVolumes
        self.totalVolumes = totalVolumes

class Mask(object):
    """Object which is a mask
    
    The masks represent different tissue types in the head. 
    The tissue types are: parenchyma (brain), haematoma, ventricle, whole head. 
    The masks can be saved across the whole head, or can be saved only on the 
    left hand side / right hand side of the head. 
    The sides are defined as being on the left and right of the midplane.
    """

    def __init__(self, midplane=None, brain=None, haematoma=None, ventricle=None, headLHS=None, headRHS=None, brainLHS=None, brainRHS=None, haematomaLHS=None, haematomaRHS=None, ventricleLHS=None, ventricleRHS=None):
        
        self.midplane    = midplane
        self.brain       = brain
        self.haematoma   = haematoma
        self.ventricle   = ventricle
        
        self.headLHS     = headLHS
        self.headRHS     = headRHS
        self.brainLHS    = brainLHS
        self.brainRHS    = brainRHS
        self.haematomaLHS   = haematomaLHS
        self.haematomaRHS   = haematomaRHS
        self.ventricleLHS   = ventricleLHS
        self.ventricleRHS   = ventricleRHS
        

