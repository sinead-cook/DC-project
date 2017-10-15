This code takes in inputs of CT scans and outputs information about that CT scan and masks. It is optimised to work with intact (before craniotomy or craniectomy) CT scans with acute subdural haematomas. It will not pick up multiple disconnected bleeds.

Information about scripts:
midplanefinder.ipynb is an interactive way of finding the midplane of a scan.
midplanefinderbatch.ipynb works the same as midplanefinder but can process multiple scans
symmetryanalysis.ipynb

To build the mac app:
  1. Clone the repository
  2. Set up an environment so that you can run 'startupdlg.py' on your computer. The easiest way to do this is to download          anaconda, and create a conda environment. The most unusual dependencies are:
     simpleITK, numpy, matplotlib, nibabel, opencv 2.4 (conda install -c menpo opencv=2.4.11) (make sure you configure this        last).
  3. Download pyinstaller into your environment
  4. Open a terminal window, activate your environment and go to the folder where the repository is located. Type in                'pyinstaller startupdlg.spec' to build the app.
  5. The app will be located in 'dist'
