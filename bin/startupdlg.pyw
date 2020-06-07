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
import master
import inspect

class StartupDlg(QDialog,
        ui_startupdlg.Ui_Dialog, master.master):

    def __init__(self, fopen=None, parent=None):

        # import sys
        # app = QApplication(sys.argv)
        # app.setQuitOnLastWindowClosed(True)
        # form = StartupDlg(0)
        # form.show()
        # app.exec_()
        
        super(StartupDlg, self).__init__(parent)
        self.setupUi(self) # setup the interface
        self.setup()       # setup the master variables
        self.cancel.clicked.connect(self.close)

        for k,v in self.__dict__.items():
            if type(v) == QCheckBox:
                checkbox_func = getattr(self,'on_StateChange_' + str(k))
                getattr(self, k).stateChanged.connect(checkbox_func)

        # setup initial method bools
        print 'setting up initial bools'
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
        else:
            return True

    def on_start_clicked(self):
        # this is where the main body of the code is run
        numiters = self.scanList.count()
        for i in range(numiters):
            self.scan.path = str(self.scanList.item(i).text())
            if self.methods.symmetry == True:
                print 'yes'
                self.analyse() # need the midplane for symmetry analysis but not for the {segmentation only} option
            self.volumeAnalysis()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    form = StartupDlg(0)
    form.show()
    app.exec_()

