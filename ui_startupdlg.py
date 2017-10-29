# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/Sinead/brain-analyser/startupdlg.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(575, 700)
        Dialog.setMinimumSize(QtCore.QSize(575, 700))
        Dialog.setMaximumSize(QtCore.QSize(575, 700))
        self.layoutWidget = QtGui.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 531, 661))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.label = QtGui.QLabel(self.layoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_4.addWidget(self.label)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.removeDuplicates = QtGui.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(12)
        self.removeDuplicates.setFont(font)
        self.removeDuplicates.setFocusPolicy(QtCore.Qt.TabFocus)
        self.removeDuplicates.setObjectName(_fromUtf8("removeDuplicates"))
        self.gridLayout.addWidget(self.removeDuplicates, 1, 2, 1, 1)
        self.addScans = QtGui.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(12)
        self.addScans.setFont(font)
        self.addScans.setObjectName(_fromUtf8("addScans"))
        self.gridLayout.addWidget(self.addScans, 1, 0, 1, 1)
        self.removeScans = QtGui.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(12)
        self.removeScans.setFont(font)
        self.removeScans.setObjectName(_fromUtf8("removeScans"))
        self.gridLayout.addWidget(self.removeScans, 1, 1, 1, 1)
        self.clearAll = QtGui.QPushButton(self.layoutWidget)
        self.clearAll.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(12)
        self.clearAll.setFont(font)
        self.clearAll.setFocusPolicy(QtCore.Qt.TabFocus)
        self.clearAll.setAutoFillBackground(False)
        self.clearAll.setStyleSheet(_fromUtf8(""))
        self.clearAll.setCheckable(False)
        self.clearAll.setAutoDefault(False)
        self.clearAll.setDefault(False)
        self.clearAll.setObjectName(_fromUtf8("clearAll"))
        self.gridLayout.addWidget(self.clearAll, 1, 3, 1, 1)
        self.start = QtGui.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(12)
        self.start.setFont(font)
        self.start.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.start.setObjectName(_fromUtf8("start"))
        self.gridLayout.addWidget(self.start, 4, 0, 1, 4)
        self.scanList = QtGui.QListWidget(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(12)
        self.scanList.setFont(font)
        self.scanList.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scanList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scanList.setAutoScroll(True)
        self.scanList.setDragEnabled(True)
        self.scanList.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        self.scanList.setObjectName(_fromUtf8("scanList"))
        self.gridLayout.addWidget(self.scanList, 0, 0, 1, 4)
        self.verticalLayout_4.addLayout(self.gridLayout)
        self.log = QtGui.QListWidget(self.layoutWidget)
        self.log.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.log.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.log.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        self.log.setVerticalScrollMode(QtGui.QAbstractItemView.ScrollPerPixel)
        self.log.setHorizontalScrollMode(QtGui.QAbstractItemView.ScrollPerPixel)
        self.log.setProperty("isWrapping", False)
        self.log.setViewMode(QtGui.QListView.ListMode)
        self.log.setObjectName(_fromUtf8("log"))
        self.verticalLayout_4.addWidget(self.log)
        self.allStop = QtGui.QPushButton(self.layoutWidget)
        self.allStop.setEnabled(False)
        self.allStop.setObjectName(_fromUtf8("allStop"))
        self.verticalLayout_4.addWidget(self.allStop)
        self.quitBtn = QtGui.QPushButton(self.layoutWidget)
        self.quitBtn.setObjectName(_fromUtf8("quitBtn"))
        self.verticalLayout_4.addWidget(self.quitBtn)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.label.setText(_translate("Dialog", "Midplane Finder", None))
        self.removeDuplicates.setText(_translate("Dialog", "Remove Duplicates", None))
        self.addScans.setText(_translate("Dialog", "Add Scans", None))
        self.removeScans.setText(_translate("Dialog", "Remove Scans", None))
        self.clearAll.setText(_translate("Dialog", "Clear All", None))
        self.start.setText(_translate("Dialog", "Start", None))
        self.allStop.setText(_translate("Dialog", "Stop all processes", None))
        self.quitBtn.setText(_translate("Dialog", "Quit", None))

