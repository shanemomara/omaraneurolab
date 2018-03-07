#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:13:31 2017

@author: Raju
"""
from PyQt5 import QtWidgets
import sys
#from imp import reload
from nc_ui import NeuroChaT_Ui

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.quitOnLastWindowClosed()
    ui= NeuroChaT_Ui()
    ui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()