from PyQt4 import QtCore, QtGui
import sys
import time
import logging
logger = logging.getLogger(__name__)

class ConsoleWindowLogHandler(logging.Handler):
    def __init__(self, textBox):
        super(ConsoleWindowLogHandler, self).__init__()
        self.textBox = textBox

    def emit(self, logRecord):
        self.textBox.append(str(logRecord.getMessage()))

def someProcess():
    logger.error("line1")
    time.sleep(5)
    logger.error("line2")

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = QtGui.QWidget()
    textBox = QtGui.QTextBrowser()
    button = QtGui.QPushButton()
    button.clicked.connect(someProcess)
    vertLayout = QtGui.QVBoxLayout()
    vertLayout.addWidget(textBox)
    vertLayout.addWidget(button)
    window.setLayout(vertLayout)
    window.show()
    consoleHandler = ConsoleWindowLogHandler(textBox)
    logger.addHandler(consoleHandler)
    sys.exit(app.exec_())

from PyQt4 import QtCore, QtGui
import sys
import time
import logging
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------
class ConsoleWindowLogHandler(logging.Handler):
    def __init__(self, sigEmitter):
        super(ConsoleWindowLogHandler, self).__init__()
        self.sigEmitter = sigEmitter

    def emit(self, logRecord):
        message = str(logRecord.getMessage())
        self.sigEmitter.emit(QtCore.SIGNAL("logMsg(QString)"), message)

#------------------------------------------------------------------------------
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()

        # Layout
        textBox = QtGui.QTextBrowser()
        self.button = QtGui.QPushButton()
        vertLayout = QtGui.QVBoxLayout()
        vertLayout.addWidget(textBox)
        vertLayout.addWidget(self.button)
        self.setLayout(vertLayout)

        # Connect button
        self.button.clicked.connect(self.buttonPressed)

        # Thread
        self.bee = Worker(self.someProcess, ())
        self.bee.finished.connect(self.restoreUi)
        self.bee.terminated.connect(self.restoreUi)

        # Console handler
        dummyEmitter = QtCore.QObject()
        self.connect(dummyEmitter, QtCore.SIGNAL("logMsg(QString)"),
                     textBox.append)
        consoleHandler = ConsoleWindowLogHandler(dummyEmitter)
        logger.addHandler(consoleHandler)

    def buttonPressed(self):
        self.button.setEnabled(False)
        self.bee.start()

    def someProcess(self):
        logger.error("starting")
        for i in xrange(10):
            logger.error("line%d" % i)
            time.sleep(2)

    def restoreUi(self):
        self.button.setEnabled(True)

#------------------------------------------------------------------------------
class Worker(QtCore.QThread):
    def __init__(self, func, args):
        super(Worker, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())