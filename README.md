# Skull Midplane Finder 
## Intro
This project is the code for my master's thesis which investigated semi-automatic midplane finder for CT scans of patients who had suffered from decompressive craniectomy. Decompressive craniectomy is an emergency surgical procedure performed on patients who have experienced severe traumatic head injury. In the procedure, a part of the skull is removed to relieve pressure on the patient's brain; pressure would have built up due to internal bleeding for decompressive craniectomy to be performed. A midline is important in medical diagnosis to determine severity of patient injury. Owing to often-unconscious patients whose heads are not upright when being scanned, midlines are difficult and time-consuming to draw in medical practice. This code uses location of the eyes in scans along with ellipse-fitting to the bone in a scan to automatically deduce a midplane in CT scans.
Masters thesis: https://www.overleaf.com/read/brvhpcbdhrgg

Code to produce midplane results is in `src` and `notebooks`. Within notebooks, `midplanefinder.ipynb` is most useful to visualise how a midplane is found. `symmetryanalysis.ipynb` will produce masks and information based on a midplane found.

The overall steps are:
<br>
<img align="center" src="https://github.com/sinead-cook/decompressive-craniectomy-midplane-finder/blob/master/readmeimages/midplane%20flowchart.png?raw=True" height="500"/>
<br>

## Set up 

To run the notebooks, CT scans of brains are required. Some examples can be found at
https://radiopaedia.org/articles/normal-brain-imaging-examples-1. Supported file extensions of scans are DICOM, NIfTI and Nrrd scans.

This repository uses python2.7, which has since been deprecated. To get the code set up, start a new python2 virtual environment and install the dependencies provided in `requirements.txt`. It is recommended that the project be forked and upgraded to python3 should a user want to do further analysis.

The binaries for an app has been built in `bin`. This app provides a user interface to the midplane finder and symmetry analysis. To build the app from scratch, install the requirements with PyQt4 and pyinstaller. Then navigate to `bin` and run `pyinstaller startupdlg.spec`. The app will be in new folder called `dist`.

## Notebooks

`midplanefinder.ipynb` provides a mask to superpose over a scan which shows the midplane of the skull. It shows the steps taken to output a midplane. `midplanefinderbatch.ipynb` contains `midplanefinder.ipynb` code without visualisation.

`symmetryanalysis.ipynb` depends of the output of `midplanefinder.ipynb` or `midplanefinderbatch.ipynb`. It outputs: 
- Paryenchyma volume (mm<sup>3</sup>) and masks either side of the midplane
- Haematoma volume (mm<sup>3</sup>) and masks either side of the mideplane
- Ventricle volume (mm<sup>3</sup>) and masks either side of the midplane
- For example:

             ['ScanB', 'w1 = 287.118632928', 'w2 = 253.708675649', 'w3 = 66',
             'Volume of haematoma on LHS is 101216.934341',
             'Volume of haematoma on RHS is 11828.987111',
             'Volume of brain on LHS is 736982.079215',
             'Volume of brain on RHS is 774164.849771',
             'Volume of CSF in ventricles on LHS is 19655.0864758',
             'Volume of CSF in ventricles on RHS is 32386.77633'],
             
w1, w2 and w3 are the x,y,z coordinates of the ventricle centroid. A user can load these coordinates into a scan viewer such as ITK snap to visualise the point. 

An example of the parenchyma, ventricle and haematoma masks from the output of `symmetryanalysis.ipynb` is:
<br>
<img src="https://github.com/sinead-cook/decompressive-craniectomy-midplane-finder/blob/master/readmeimages/outputdemonstration.png?raw=True" height="500" />
<br>



