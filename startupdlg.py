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

import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import scandata
import ui_startupdlg
import ui_quitdlg
import core 
import os 
import sys
import multiprocessing

import master
import inspect
from PyQt4.QtCore import QThread, SIGNAL


# redirect all print outs to this file
sys.stdout = open('errors.txt', 'w')
sys.stderr = open('errors.txt', 'w')

class StartupDlg(QDialog,
        ui_startupdlg.Ui_Dialog, master.master):

    def __init__(self, fopen=None, parent=None):

        # create the app window
        super(StartupDlg, self).__init__(parent)

        self.setupUi(self) # setup the interface
        self.setup()       # setup the master variables
        self.quitBtn.clicked.connect(self.close) 

        self.log.addItem(QString('Choose scans to find midplanes for and press start'))
        # global stringToPrint always contains what you want to be printed out to the logging widget in the app window
        global stringToPrint 
        global mainControl
        mainControl = 'Active'

        self.start.clicked.connect(self.start_main_code_body)

        stringToPrint = 'Choose scans to find midplanes for and press start'
        self.start_updating_log()



    # the next section controls how the addScan button works
    @pyqtSignature("")
    def on_addScans_clicked(self):
        global stringToPrint
        stringToPrint = 'Adding Scans'

        # scandata.Scan.formats (which is set in the file "scandata") controls what formats
        # the tool is currently able to analyse. Only these formats will be selectable when
        # the user is trying to add scans.
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

    # control quite variable
    def okToContinue(self): 
        reply = QMessageBox.question(self, "Quit",
                        "Are you sure you want to quit? Analysis for the current scan will be lost", QMessageBox.Yes|QMessageBox.Cancel)
        if reply == QMessageBox.Cancel:
            return False
        else: return True

    # when start is clicked, grey out some buttons and enable some others
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
        else:
            self.output_data = np.vstack((self.output_data, self.output_i))

        if i == self.scanList.count()-1:
            if self.scan.path.endswith('.nii') or self.scan.path.endswith('.nii.gz') or self.scan.path.endswith('.nrrd'):
                pathHead = os.path.split(self.scan.path)[0]
            elif self.scan.path.endswith('.dcm'):
                pathHead = os.path.split(os.path.split(self.scan.path)[0])[0]
            savePath = os.path.join(pathHead, "symmetry_analysis_results.csv")
            np.savetxt(savePath, self.output_data, delimiter=",", fmt="%s")


    def start_updating_log(self):
        self.get_thread = getPostsThread(stringToPrint)
        self.connect(self.get_thread, SIGNAL("add_post(QString)"), self.add_post)
        self.get_thread.start()

    # when start is clicked, start all the processes
    def start_main_code_body(self):
        self.get_main_thread = mainCodeThread(self.scanList)
        self.connect(self.get_main_thread, SIGNAL("readIn(QString)")         , self.readIn)
        self.connect(self.get_main_thread, SIGNAL("reshapingScan(QString)")  , self.reshapingScan)
        self.connect(self.get_main_thread, SIGNAL("findingEyes(QString)")    , self.findingEyes)
        self.connect(self.get_main_thread, SIGNAL("findingSkew(QString)")    , self.findingSkew)
        self.connect(self.get_main_thread, SIGNAL("correctSkew(QString)")    , self.correctSkew)
        self.connect(self.get_main_thread, SIGNAL("findingEyes2(QString)")   , self.findingEyes2)
        self.connect(self.get_main_thread, SIGNAL("ellipseFitting(QString)") , self.ellipseFitting)
        self.connect(self.get_main_thread, SIGNAL("findingMidplane(QString)")  , self.findingMidplane)

        self.connect(self.get_main_thread, SIGNAL("savingMasks(QString)")       , self.savingMasks)
    
        self.get_main_thread.start()


    def finishedProcesses(self):
        self.start.setEnabled(True)
        self.allStop.setEnabled(False)

        self.addScans.setEnabled(True)
        self.removeScans.setEnabled(True)
        self.removeDuplicates.setEnabled(True)
        self.clearAll.setEnabled(True)
        self.get_main_thread.terminate()


    def add_post(self, post_text):
        previousMessage = self.log.item(int(self.log.count())-1).text()
 
        if previousMessage != post_text:
            self.log.addItem(QString(post_text))
            self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())


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
    def __init__(self, scanList):
        QThread.__init__(self)
        global mainControl 
        self.mainControl = mainControl
        self.scanList = scanList

    def run(self):

        numiters = self.scanList.count()

        for i in range(numiters):
            global stringToPrint
            stringToPrint = 'Reading in scan number %d' % (i+1)

            self.sleep(5)

            self.path = str(self.scanList.item(i).text())

            # global stringToPrint
            self.emit(SIGNAL('readIn(QString)'), self.path)
            stringToPrint = 'Scan read in. Start reshaping scans'
            self.sleep(10)

            # global stringToPrint
            self.emit(SIGNAL('reshapingScan(QString)'), self.path)
            stringToPrint = 'Reshaping complete. Start finding eyes'
            self.sleep(10)

            self.emit(SIGNAL('findingEyes(QString)'), self.path)
            stringToPrint = 'Eyes found. Start finding skews'
            self.sleep(10)
            
            self.emit(SIGNAL('findingSkew(QString)'), self.path)
            stringToPrint = 'Skews found. Start correcting skews'
            self.sleep(10)

            self.emit(SIGNAL('correctSkew(QString)'), self.path)
            stringToPrint = 'Skews corrected. Start finding eyes in corrected scan'
            self.sleep(10)
            
            self.emit(SIGNAL('findingEyes2(QString)'), self.path)
            stringToPrint = 'Eyes found. Start fitting ellipses'
            self.sleep(10)

            self.emit(SIGNAL('ellipseFitting(QString)'), self.path)
            stringToPrint = 'Ellipses fitted. Start finding midplane'
            self.sleep(10)

            self.emit(SIGNAL('findingMidplane(QString)'), self.path)
            self.sleep(10)

            self.emit(SIGNAL('savingMasks(QString)'), self.path)

            if self.path.endswith('.nii') or self.path.endswith('.nii.gz') or self.path.endswith('.nrrd'):
                pathHead = os.path.split(self.path)[0]
            elif self.path.endswith('.dcm'):
                pathHead = os.path.split(os.path.split(self.path)[0])[0]
            stringToPrint = 'Midplane found. Midplane mask saved in %s' % pathHead
            self.sleep(10)

        self.mainControl = 'Finished'
        self.emit(SIGNAL('finish(QString)'), self.mainControl)

if __name__ == "__main__":
    import sys
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    form = StartupDlg(0)
    form.show()
    form.raise_()
    app.exec_()

