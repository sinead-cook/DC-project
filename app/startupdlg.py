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
import core
import master

class StartupDlg(QDialog,
        ui_startupdlg.Ui_Dialog, master.master):

    def __init__(self, fopen=None, parent=None):
        super(StartupDlg, self).__init__(parent)
        self.setupUi(self)

    @pyqtSignature("")
    def on_addScans_clicked(self):
        self.setup()
        self.scan.path = str(QFileDialog.getOpenFileName(self, filter=scandata.Scan.formats()))
        self.analyse()
        # the open thing wasn't working because the two things have different backends
        # trying using self.(master modules) again because didn't try it properly


        # if fopen is not None:
        #     self.fileOpen

        # self.pushButtonSelectScan.clicked.connect(self.fileOpen)
        # self.pushButtonGo.clicked.connect(self.analysis)
                                                    

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    form = StartupDlg(0)
    form.show()
    app.exec_()

