# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/Sinead/app/quitdlg.ui'
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

class Ui_quitdlg(object):
    def setupUi(self, quitdlg):
        quitdlg.setObjectName(_fromUtf8("quitdlg"))
        quitdlg.resize(275, 110)
        quitdlg.setMinimumSize(QtCore.QSize(275, 110))
        quitdlg.setMaximumSize(QtCore.QSize(275, 110))
        self.layoutWidget = QtGui.QWidget(quitdlg)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 231, 77))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setWordWrap(True)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.buttonBox = QtGui.QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Yes)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(quitdlg)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), quitdlg.close)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), quitdlg.close)
        QtCore.QMetaObject.connectSlotsByName(quitdlg)

    def retranslateUi(self, quitdlg):
        quitdlg.setWindowTitle(_translate("quitdlg", "Dialog", None))
        self.label.setText(_translate("quitdlg", "Are you sure you want to quit? ", None))

