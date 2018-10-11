#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Use this module to invoke the graphical user interface of NeuroChaT.

Run this code in command line or from IDE and it will start the graphical window
of the NeuroChaT software.

@author: Md Nurul Islam; islammn at tcd dot ie
"""

import sys
from PyQt5 import QtWidgets

sys.path.insert(1, 'C:\\Users\\Raju\\Google Drive\\NeuroChaT Py\\neurochat\\')

from neurochat.nc_ui import NeuroChaT_Ui

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.quitOnLastWindowClosed()
    ui= NeuroChaT_Ui()
    ui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
