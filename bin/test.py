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
from PyQt4.QtGui import *
import scandata
import ui_startupdlg
import ui_quitdlg
import core
import os 
import sys
import multiprocessing
import inspect
from PyQt4.QtCore import QThread, SIGNAL
import master

from os.path import expanduser
home = expanduser("~")

try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    base_path = sys._MEIPASS2
except Exception:
    base_path = os.path.abspath(".")

sys.path.append(base_path)

os.chdir(home)

# import sys
sys.stdout = open('stdout.txt', 'w')
sys.stderr = open('stdout.txt', 'w')

class StartupDlg(QDialog,
        ui_startupdlg.Ui_Dialog, master.master):

    def __init__(self, fopen=None, parent=None):

        super(StartupDlg, self).__init__(parent)

        self.setupUi(self) # setup the interface
        self.setup()       # setup the master variables
        self.quitBtn.clicked.connect(self.close)

        self.log.addItem(QString('Select desired options, choose scans to analyse and press start'))
        global stringToPrint
        global mainControl
        mainControl = 'Active'
        self.start.clicked.connect(self.start_main_code_body)

        mainControl = 'Active'
        stringToPrint = 'Select desired options, choose scans to analyse and press start'
        self.start_updating_log()


        for k,v in self.__dict__.items():
            if type(v) == QCheckBox:
                checkbox_func = getattr(self,'on_StateChange_' + str(k))
                getattr(self, k).stateChanged.connect(checkbox_func)
        # setup initial method bools
        # print 'setting up initial bools'

        self.methods.parenchyma = True
        self.methods.haematoma  = True
        self.methods.ventricles = True
        self.methods.wholeBrain = True
        self.methods.symmetry   = True
        self.methods.segmentation = False

        # setupt original outputs to save
        self.methods.totalSymmVolumes = True
        self.methods.totalVolumes  = True
        self.methods.tissueMasks   = True
        self.methods.lrTissueMasks = True
        self.methods.midplaneMask  = False

    def on_StateChange_haematoma(self):
        if self.haematoma.isChecked():
            self.methods.haematoma = True
        else:
            self.methods.haematoma = False

    def on_StateChange_parenchyma(self):
        if self.parenchyma.isChecked():
            self.methods.parenchyma = True
        else:
            self.methods.parenchyma = False

    def on_StateChange_segmentation(self):
        if self.segmentation.isChecked():
            self.symmetry.setChecked(False)
            self.totalSymmVolumes.setEnabled(False)
            self.totalSymmVolumes.setChecked(False)
            self.lrTissueMasks.setEnabled(False)
            self.lrTissueMasks.setChecked(False)
            self.midplaneMask.setEnabled(False)
            self.midplaneMask.setChecked(False)

            self.methods.segmentation = True

        else:
            self.symmetry.setChecked(True)
            self.totalSymmVolumes.setEnabled(True)
            self.lrTissueMasks.setEnabled(True)
            self.midplaneMask.setEnabled(True)

            self.methods.segmentation = False

    def on_StateChange_symmetry(self):
        if self.symmetry.isChecked():
            self.segmentation.setChecked(False)

            self.methods.symmetry = True

        else:
            self.segmentation.setChecked(True)

            self.methods.symmetry = False

    def on_StateChange_ventricles(self):
        if self.ventricles.isChecked():
            self.methods.ventricles = True
        else:
            self.methods.ventricles = False

    def on_StateChange_wholeBrain(self):
        if self.wholeBrain.isChecked():
            self.methods.wholeBrain = True
        else:
            self.methods.wholeBrain = False

    def on_StateChange_lrTissueMasks(self):
        if self.lrTissueMasks.isChecked():
            self.methods.lrTissueMasks = True
        else:
            self.methods.lrTissueMasks = False

    def on_StateChange_midplaneMask(self):
        if self.midplaneMask.isChecked():
            self.methods.midplaneMask = True
        else:
            self.methods.midplaneMask = False

    def on_StateChange_tissueMasks(self):
        if self.tissueMasks.isChecked():
            self.methods.tissueMasks = True
        else:
            self.methods.tissueMasks = False

    def on_StateChange_totalSymmVolumes(self):
        if self.totalSymmVolumes.isChecked():
            self.methods.totalSymmVolumes = True
        else:
            self.methods.totalSymmVolumes = False

    def on_StateChange_totalVolumes(self):
        if self.totalVolumes.isChecked():
            self.methods.totalVolumes = True
        else:
            self.methods.totalVolumes = False

    @pyqtSignature("")
    def on_addScans_clicked(self):
        global stringToPrint
        stringToPrint = 'Adding Scans'
        clicked_paths = QFileDialog.getOpenFileNames(self, filter=scandata.Scan.formats())
        self.scanList.addItems(clicked_paths)

    def on_removeDuplicates_clicked(self):
        index = 1
        numiters = self.scanList.count()
        for index in xrange(numiters):
            if index>=self.scanList.count():
                break
            current_item = self.scanList.item(index)
            if len(self.scanList.findItems(current_item.text(), Qt.MatchExactly))>1:
                self.scanList.takeItem(index)

    def on_removeScans_clicked(self):
        rows=[idx.row() for idx in self.scanList.selectedIndexes()]
        for i in range(len(rows)):
            n = rows[i]
            self.scanList.takeItem(n)

    def on_clearAll_clicked(self):
        for index in xrange(self.scanList.count()):
            self.scanList.takeItem(index)

    def closeEvent(self, event):
        if not self.okToContinue():
            event.ignore()


    def okToContinue(self): # from chapter 13, python editor
        reply = QMessageBox.question(self, "Quit",
                        "Are you sure you want to quit? All settings will be lost", QMessageBox.Yes|QMessageBox.Cancel)
        if reply == QMessageBox.Cancel:
            return False
        else: return True

    def on_start_clicked(self):
        numiters = self.scanList.count()
        global stringToPrint
        if numiters == 0: 
            stringToPrint = 'Please select scans to analyse'
            return
        self.start.setEnabled(False)
        self.allStop.setEnabled(True)
        self.addScans.setEnabled(False)
        self.removeScans.setEnabled(False)
        self.removeDuplicates.setEnabled(False)
        self.clearAll.setEnabled(False)


    def on_allStop_clicked(self):
        reply = QMessageBox.question(self, "Interrupt", "Are you sure you want to stop analysis? All processes for the current scan will be lost", QMessageBox.Yes|QMessageBox.Cancel)
        if reply == QMessageBox.Cancel:
            return
        else:
            self.get_main_thread.terminate()
            self.get_thread.terminate()
            self.start.setEnabled(True)
            self.allStop.setEnabled(False)

            self.addScans.setEnabled(True)
            self.removeScans.setEnabled(True)
            self.removeDuplicates.setEnabled(True)
            self.clearAll.setEnabled(True)

            global stringToPrint
            stringToPrint = 'Processes interrupted'
            self.start_updating_log()
            return

    def saveOutputData(self, i):
        import numpy as np
        if i == 0:
            self.output_data = ['Scan Path', 'Left haematoma volume', 'Right haematoma volume', 'Left ventricular volume', 'Right ventricular volume', 'Left brain volume', 'Right brain volume', 'Left parenchymal volume', 'Right parenchymal volume', 'Total haematoma volume', 'Total ventricular volume', 'Total brain volume', 'Total parenchymal volume']
            self.output_data = np.vstack((self.output_data, self.output_i))
            # print self.output_i
        else:
            self.output_data = np.vstack((self.output_data, self.output_i))

        if i == self.scanList.count()-1:
            if self.scan.path.endswith('.nii') or self.scan.path.endswith('.nii.gz') or self.scan.path.endswith('.nrrd'):
                pathHead = os.path.split(self.scan.path)[0]
            elif self.scan.path.endswith('.dcm'):
                pathHead = os.path.split(os.path.split(self.scan.path)[0])[0]
            savePath = os.path.join(pathHead, "symmetry_analysis_results.csv")
            np.savetxt(savePath, self.output_data, delimiter=",", fmt="%s")


    def testBET(self):
        # BET FSL
        import subprocess
        import nibabel as nib
        import numpy as np

        img = nib.Nifti1Image(self.scan.array, np.eye(4))
        if self.scan.path.endswith('.nii') or self.scan.path.endswith('.nii.gz'):
            pathHead = os.path.split(self.scan.path)[0]
        elif self.scan.path.endswith('nrrd'):
            pathHead = os.path.split(self.scan.path)[0]
        elif self.scan.path.endswith('.dcm'):
            pathHead = os.path.split(os.path.split(self.scan.path)[0])[0]
        tempfile = os.path.join(pathHead, 'temp.nii.gz')
        outfile = os.path.join(pathHead, 'out.nii.gz')
        img = nib.Nifti1Image(self.scan.array, np.eye(4))
        nib.save(img, tempfile)

        outputFileName = 'betlog.txt'
        outputFile = open(outputFileName, "w")
        originalcwd = os.getcwd()
        p = subprocess.Popen(['which bet'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
        result = p.stdout.read()
        os.environ['PATH'] += os.path.split(result)[0]
        print os.path.split(result)[0]
        print 'bet {} {} -f 0.3'.format(tempfile, outfile)
        os.chdir(result)
        command = ['bet {} {} -f 0.3'.format(tempfile, outfile)]
        proc = subprocess.Popen(command, cwd=os.path.split(result)[0], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
        proc.stdin.close()
        proc.wait()
        result = proc.returncode
        os.chdir(originalcwd)
        outputFile.write(proc.stdout.read())

        print 'FSL BET tool complete'


    def start_updating_log(self):
        self.get_thread = getPostsThread(stringToPrint)
        self.connect(self.get_thread, SIGNAL("add_post(QString)"), self.add_post)
        self.get_thread.start()

    def start_main_code_body(self):
        self.get_main_thread = mainCodeThread(self.methods, self.scanList)
        self.connect(self.get_main_thread, SIGNAL("readIn(QString)")         , self.readIn)
        self.connect(self.get_main_thread, SIGNAL("reshapingScan(QString)")  , self.reshapingScan)
        self.connect(self.get_main_thread, SIGNAL("findingEyes(QString)")    , self.findingEyes)
        self.connect(self.get_main_thread, SIGNAL("findingSkew(QString)")    , self.findingSkew)
        self.connect(self.get_main_thread, SIGNAL("correctSkew(QString)")    , self.correctSkew)
        self.connect(self.get_main_thread, SIGNAL("findingEyes2(QString)")   , self.findingEyes2)
        self.connect(self.get_main_thread, SIGNAL("ellipseFitting(QString)") , self.ellipseFitting)
        self.connect(self.get_main_thread, SIGNAL("findingMidplane(QString)")  , self.findingMidplane)

        self.connect(self.get_main_thread, SIGNAL("savingMasks(QString)")       , self.savingMasks)
        self.connect(self.get_main_thread, SIGNAL("volumeAnalysis(QString)")    , self.volumeAnalysis)
        self.connect(self.get_main_thread, SIGNAL("skinOrbital(QString)")       , self.skinOrbital)
        self.connect(self.get_main_thread, SIGNAL("brainExtraction(QString)")   , self.brainExtraction)
        self.connect(self.get_main_thread, SIGNAL("findingVentricles(QString)") , self.findingVentricles)
        self.connect(self.get_main_thread, SIGNAL("findingHaematoma(QString)")  , self.findingHaematoma)
        self.connect(self.get_main_thread, SIGNAL("savingExtraMasks(QString)")  , self.savingExtraMasks)

        self.connect(self.get_main_thread, SIGNAL("add_post(QString)")          , self.add_post)
        self.connect(self.get_main_thread, SIGNAL('finish(QString)')            , self.finishedProcesses)

        self.connect(self.get_main_thread, SIGNAL('saveOutputData(int)')        , self.saveOutputData)

        self.connect(self.get_main_thread, SIGNAL('testBET(int)')        , self.testBET)

        self.get_main_thread.start()


    def finishedProcesses(self):
        self.start.setEnabled(True)
        self.allStop.setEnabled(False)

        self.addScans.setEnabled(True)
        self.removeScans.setEnabled(True)
        self.removeDuplicates.setEnabled(True)
        self.clearAll.setEnabled(True)
        self.get_main_thread.terminate()

        # self.start.clicked.connect(self.get_main_thread.start(self.methods, self.scanList))

    def add_post(self, post_text):
        previousMessage = self.log.item(int(self.log.count())-1).text()
        # self.log.verticalScrollBar().setValue(
        # self.log.verticalScrollBar().maximum())
        if previousMessage != post_text:
            self.log.addItem(QString(post_text))
            self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())
            QApplication.processEvents()


class getPostsThread(QThread):
    def __init__(self, messages):
        QThread.__init__(self)
        global stringToPrint
        self.messages = stringToPrint
    def run(self):
        while True:
            try:
                self.messages = stringToPrint
                self.emit(SIGNAL('add_post(QString)'), self.messages)
                self.sleep(1)
            except:
                continue


class mainCodeThread(QThread):
    import os
    def __init__(self, methods, scanList):
        QThread.__init__(self)
        global mainControl 
        self.mainControl = mainControl
        self.methods = methods
        self.scanList = scanList

    def run(self):
        
        # mainControl = 'Active'
            # while mainControl = True:
        #     print 'seeing whether 2 threads run at same time'
        #     time.sleep(1)
        numiters = self.scanList.count()
        # print numiters
        for i in range(numiters):


            global stringToPrint
            stringToPrint = 'Reading in scan number %d' % (i+1)

            self.sleep(5)

            self.path = str(self.scanList.item(i).text())
            if self.methods.symmetry == True:

                # global stringToPrint
                self.emit(SIGNAL('readIn(QString)'), self.path)
                stringToPrint = 'Scan read in. Start reshaping scans'
                self.sleep(10)
                self.emit(SIGNAL('testBET(int)'), 5)


            #     # global stringToPrint
            #     self.emit(SIGNAL('reshapingScan(QString)'), self.path)
            #     stringToPrint = 'Reshaping complete. Start finding eyes'
            #     self.sleep(10)

            #     self.emit(SIGNAL('findingEyes(QString)'), self.path)
            #     stringToPrint = 'Eyes found. Start finding skews'
            #     self.sleep(10)
                
            #     self.emit(SIGNAL('findingSkew(QString)'), self.path)
            #     stringToPrint = 'Skews found. Start correcting skews'
            #     self.sleep(10)

            #     self.emit(SIGNAL('correctSkew(QString)'), self.path)
            #     stringToPrint = 'Skews corrected. Start finding eyes in corrected scan'
            #     self.sleep(10)
                
            #     self.emit(SIGNAL('findingEyes2(QString)'), self.path)
            #     stringToPrint = 'Eyes found. Start fitting ellipses (long process)'
            #     self.sleep(10)

            #     self.emit(SIGNAL('ellipseFitting(QString)'), self.path)
            #     stringToPrint = 'Ellipses fitted. Start finding midplane'
            #     self.sleep(10)

            #     self.emit(SIGNAL('findingMidplane(QString)'), self.path)
            #     stringToPrint = 'Midplane found.'
            #     self.sleep(10)

            # self.emit(SIGNAL('savingMasks(QString)'), self.path)
            # if self.methods.symmetry == False:
            #     stringToPrint = 'Read in scan and start volume analysis'
            # else:
            #     stringToPrint = 'Midplane found. Start volume analysis.'
            # self.sleep(10)

            # self.emit(SIGNAL('volumeAnalysis(QString)'), self.path)
            # stringToPrint = 'Skin and orbital masks found.'
            # self.sleep(10)

            # self.emit(SIGNAL('skinOrbital(QString)'), self.path)
            # stringToPrint = 'Skin and orbital masks found.'
            # self.sleep(10)

            # self.emit(SIGNAL('brainExtraction(QString)'), self.path)
            # stringToPrint = 'Skin and orbital masks found.'
            # self.sleep(10) 

            # self.emit(SIGNAL('findingVentricles(QString)'), self.path)
            # if self.methods.ventricles == True:
            #     stringToPrint = 'Ventricles mask found'
            # self.sleep(10)

            # self.emit(SIGNAL('findingHaematoma(QString)'), self.path)
            # if self.methods.haematoma == True:
            #     stringToPrint = 'haematoma mask found'
            # self.sleep(10)

            # self.emit(SIGNAL('savingExtraMasks(QString)'), self.path)
            # stringToPrint = 'Extra masks saved. Volumes analysis saved'
            # self.sleep(10)

            # if self.path.endswith('.nii') or self.path.endswith('.nii.gz') or self.path.endswith('.nrrd'):
            #     pathHead = os.path.split(self.path)[0]
            # elif self.path.endswith('.dcm'):
            #     pathHead = os.path.split(os.path.split(self.path)[0])[0]
            # stringToPrint = 'All masks saved in %s' % pathHead
            # self.sleep(10)
            
            # if i == numiters-1:
            #     stringToPrint = 'All symmetry analysis data saved in %s' % pathHead

            # self.emit(SIGNAL('saveOutputData(int)'), i)

        self.mainControl = 'Finished'
        self.emit(SIGNAL('finish(QString)'), self.mainControl)


if __name__ == "__main__":
    import sys
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    form = StartupDlg(0)
    form.show()
    app.exec_()

