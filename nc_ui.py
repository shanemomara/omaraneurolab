# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'NeuroChaT.ui'
#
# Created: Thu Jan 12 13:04:12 2017
#      by: PyQt4 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!
from PyQt5 import QtCore, QtWidgets, QtGui
from imp import reload
import nc_data
reload(nc_data)
from nc_data import Nhdf
import nc_ext
reload(nc_ext)
from nc_ext import *
import nc_control
reload(nc_control)
from nc_control import NeuroChaT
import logging
import pandas as pd
import os
import sys
from functools import partial
import shutil

from PyPDF2 import PdfFileReader, PdfFileMerger

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s
try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)    
 
class NOut(QtCore.QObject):
    emitted= QtCore.pyqtSignal(str)
    def __init__(self):    
        super().__init__()
    def write(self, text):
#        self.emit(QtCore.SIGNAL('update_log(QString)'), text) 
        self.emitted.emit(text)
        
class NeuroChaT_Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        
        self.nout= NOut()
        sys.stdout.write=  self.nout.write
        self._control= NeuroChaT(parent= self)
        self._resultsUi= Ui_results(self)
        self._mode_dict= self._control.getAllModes()
        self._currDir= "/home/"
        self.setupUi()
        
    def setupUi(self):
        self.setObjectName(_fromUtf8("MainWindow"))
        self.setEnabled(True)
        self.setFixedSize(725, 420)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        
        layer_6_1= QtWidgets.QVBoxLayout()
        self.modeLabel= addLabel_2("Analysis Mode", "modeLabel")        
        self.modeBox= addComboBox_2("modeBox")
        layer_6_1.addWidget(self.modeLabel)
        layer_6_1.addWidget(self.modeBox)

        layer_6_2= QtWidgets.QVBoxLayout()
        self.unitLabel= addLabel_2("Unti No", "unitLabel")
        self.unitNoBox= addComboBox_2("unitNoBox")
        self.unitNoBox.addItems([str(i) for i in list(range(256))])
#        self.unitNoBox.setEditable(True)
        layer_6_2.addWidget(self.unitLabel) 
        layer_6_2.addWidget(self.unitNoBox)
        
        layer_6_3= QtWidgets.QVBoxLayout()
        self.chanLabel= addLabel_2("LFP Ch No", "chanLabel")
        self.lfpChanBox= addComboBox_2("lfpChanBox")
        self.lfpChanBox.setSizeAdjustPolicy(self.lfpChanBox.AdjustToContents)
#        self.lfpChanBox.setEditable(True)
        layer_6_3.addWidget(self.chanLabel)
        layer_6_3.addWidget(self.lfpChanBox)

        layer_5_1= QtWidgets.QHBoxLayout()
        layer_5_1.addLayout(layer_6_1)
#        layer_5_1.addStretch(1)
        layer_5_1.addLayout(layer_6_2)
        layer_5_1.addLayout(layer_6_3)
        
        layer_5_2= QtWidgets.QHBoxLayout()
        self.browseButton = addPushButton_2("Browse", "browseButton")
        self.filenameLine = addLineEdit_2("Select spike(.n) &/or position file(.txt)", "filenameLine")
        layer_5_2.addWidget(self.browseButton)
        layer_5_2.addWidget(self.filenameLine)
        
        layer_4_1= QtWidgets.QVBoxLayout()
        layer_4_1.addLayout(layer_5_1)
        layer_4_1.addLayout(layer_5_2)
        
        self.cellTypeBox= self.selectCellTypeUi()
        layer_3_2= QtWidgets.QVBoxLayout()
        layer_3_2.addLayout(layer_4_1)
        layer_3_2.addWidget(self.cellTypeBox)

        self.inpFormatLabel= addLabel_2("Input Data Format", "inpFormatLabel")
        self.fileFormatBox= addComboBox_2("fileFormatBox")
        self.graphicFormatBox= self.selectGraphicFormatUi()
        self.startButton= addPushButton_2("Start", "startButton")
        self.saveLogButton = addPushButton_2("Save log", "saveLogButton")
        self.clearLogButton = addPushButton_2("Clear log", "clearLogButton")
        layer_3_1= QtWidgets.QVBoxLayout()
        layer_3_1.addWidget(self.inpFormatLabel)
        layer_3_1.addWidget(self.fileFormatBox)
        layer_3_1.addWidget(self.graphicFormatBox)
        layer_3_1.addStretch(1)
        self.startButton.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        layer_3_1.addWidget(self.startButton)
        layer_3_1.addStretch(1)
        
        layer_2_1= QtWidgets.QHBoxLayout()
        layer_2_1.addLayout(layer_3_1)
        layer_2_1.addLayout(layer_3_2, 2)
        
        layer_2_2= QtWidgets.QHBoxLayout()
        self.saveLogButton.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.clearLogButton.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        layer_2_2.addWidget(self.saveLogButton, 0, QtCore.Qt.AlignLeft)
        layer_2_2.addWidget(self.clearLogButton, 0, QtCore.Qt.AlignRight)
        
        self.logText = addLogBox("logText")
        layer_1_1= QtWidgets.QVBoxLayout()
        layer_1_1.addLayout(layer_2_1)
        layer_1_1.addLayout(layer_2_2)
        layer_1_1.addWidget(self.logText)
        
        layer_1_2= self.selectAnalysisUi()
        
        final_layer= QtWidgets.QHBoxLayout()
        final_layer.addLayout(layer_1_1, 4)
        final_layer.addLayout(layer_1_2, 2)
        
#        self.modeBox.addItems(["Single Unit", "Single Session", "Listed Units", "Multiple Sessions"])
        self.modeBox.addItems(["Single Unit", "Single Session", "Listed Units"])
        self.fileFormatBox.addItems(["Axona", "Neuralynx", "NWB"])
        self.lfpChan_getitems()

        
        self.centralwidget.setLayout(final_layer)

        self.menuUi()        
        self.retranslateUi()
        
        # Set up child windows
        self._resultsUi.setupUi()
        self._mergeUi= Ui_merge(self)        
        self._paramUi= Ui_parameters(self)
                
        self.setCentralWidget(self.centralwidget)        
        QtCore.QMetaObject.connectSlotsByName(self)
        
        # Set the callbacks
        self.behaviourUi()
        
    def behaviourUi(self):
#       self.connect(self.nout, QtCore.SIGNAL('update_log(QString)'), self.update_log)
       self.nout.emitted[str].connect(self.update_log)        
       self.fileFormatBox.currentIndexChanged[int].connect(self.data_format_select)       
       self.modeBox.currentIndexChanged[int].connect(self.mode_select)
       self.pdfButton.setChecked(True)
       self.graphicFormatGroup.buttonClicked.connect(self.graphic_format_select)
       self.unitNoBox.currentIndexChanged[int].connect(self.set_unit_no)
       self.lfpChanBox.currentIndexChanged[int].connect(self.set_lfp_chan)
       self.selectAll.stateChanged.connect(self.select_all)
       self.browseButton.clicked.connect(self.browse)
       self.clearLogButton.clicked.connect(self.clear_log)
       self.saveLogButton.clicked.connect(self.save_log)
        
       self.openFileAct.triggered.connect(self.browse)
       self.saveSessionAct.triggered.connect(self.save_session)
       self.loadSessionAct.triggered.connect(self.load_session)

       self.startButton.clicked.connect(self.start)
              
       self.exitAct.triggered.connect(self.exit_nc)

       self.exportResultsAct.triggered.connect(self.export_results)
       self.exportGraphicInfoAct.triggered.connect(self.export_graphicInfo)

       self.mergeAct.triggered.connect(self.merge_output)
       self.accumulateAct.triggered.connect(self.accumulate_output)


       self.verifyUnitsAct.triggered.connect(self.verify_units)
#       self.evaluateAct.triggered.connect(self.evaluate_clustering)        
       self.compareUnitsAct.triggered.connect(self.compare_units)
       self.convertFilesAct.triggered.connect(self.convert_to_nwb)
       self.paramSetAct.triggered.connect(self.set_parameters)        
        
       self._resultsUi.exportButton.clicked.connect(self.export_results)       
       
       self.cellTypeGroup.buttonClicked.connect(self.cell_type_select)
        
    def menuUi(self):
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 722, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.fileMenu= self.menubar.addMenu('&File')
        self.settingsMenu= self.menubar.addMenu('&Settings')
        self.utilitiesMenu= self.menubar.addMenu('&Utilities')
        self.helpMenu= self.menubar.addMenu('&Help')
                
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        self.setStatusBar(self.statusbar)

        self.openFileAct= self.fileMenu.addAction("Open...")
        self.openFileAct.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        
        self.fileMenu.addSeparator()
        
        self.saveSessionAct= self.fileMenu.addAction("Save session...")
        self.saveSessionAct.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        
        self.loadSessionAct= self.fileMenu.addAction("Load session...")
        self.loadSessionAct.setShortcut(QtGui.QKeySequence("Ctrl+L"))

        self.fileMenu.addSeparator()
    
        self.exitAct= self.fileMenu.addAction("Exit")
        self.exitAct.setShortcut(QtGui.QKeySequence("Ctrl+Q"))

        self.paramSetAct= self.settingsMenu.addAction("Parameters")
        self.paramSetAct.setShortcut(QtGui.QKeySequence("Ctrl+P"))

        self.exportResultsAct= self.utilitiesMenu.addAction("Export results")
        self.exportResultsAct.setShortcut(QtGui.QKeySequence("Ctrl+T"))
        
        self.exportGraphicInfoAct= self.utilitiesMenu.addAction("Export graphic file info")
        self.exportGraphicInfoAct.setShortcut(QtGui.QKeySequence("Ctrl+G"))
        
        self.utilitiesMenu.addSeparator()
        
        self.mergeAct= self.utilitiesMenu.addAction("Merge Output PS/PDF")
        self.accumulateAct= self.utilitiesMenu.addAction("Accumulate output PS/PDF")
        
        self.utilitiesMenu.addSeparator()

        self.verifyUnitsAct= self.utilitiesMenu.addAction("Verify units")
        self.evaluateAct= self.utilitiesMenu.addAction("Evaluate clustering")        
        self.compareUnitsAct= self.utilitiesMenu.addAction("Compare single units")
        self.convertFilesAct= self.utilitiesMenu.addAction("Convert to NWB format")

        self.viewHelpAct= self.helpMenu.addAction("NeuroChaT documentation")
        self.viewHelpAct.setShortcut(QtGui.QKeySequence("F1"))
        self.tutorialAct= self.helpMenu.addAction("NeuroChaT tutorial")
        self.helpMenu.addSeparator()
        self.aboutNCAct= self.helpMenu.addAction("About NeuroChaT")
    
    def selectGraphicFormatUi(self):        
       
        self.pdfButton = addRadioButton_2("PDF", "pdfButton")
        self.psButton = addRadioButton_2("Postscript", "psButton")

        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.pdfButton)
        layout.addWidget(self.psButton)
        
        graphicFormatBox = addGroupBox_2("Graphic Format", "graphicFormatBox")        
        graphicFormatBox.setLayout(layout)
        
        self.graphicFormatGroup = QtWidgets.QButtonGroup(graphicFormatBox)
        self.graphicFormatGroup.setObjectName(_fromUtf8("graphicFormatGroup"))        
        self.graphicFormatGroup.addButton(self.pdfButton)
        self.graphicFormatGroup.addButton(self.psButton)
        
        return graphicFormatBox
        
    def selectCellTypeUi(self):        
        positions= [(i, j) for j in range(3) for i in range(4)]        

        self.placeCellButton = addRadioButton_2("Place", "placeCell")

        self.hdCellButton = addRadioButton_2("Head-directional", "hdCellButton")
        
        self.gridCellButton = addRadioButton_2("Grid", "gridCellButton")
        
        self.boundaryCellButton = addRadioButton_2("Boundary", "boundaryCellButton")
        
        self.gradientCellButton = addRadioButton_2("Gradient", "gradientCellButton")
        
        self.hdXPlaceCellButton = addRadioButton_2("HDxPlace", "hdXPlaceCellButton")

        self.thetaCellButton = addRadioButton_2("Theta-rhythmic", "thetaCellButton")
        
        self.thetaSkipCellButton = addRadioButton_2("Theta-skipping", "thetaSkipCellButton")

        layout= QtWidgets.QGridLayout()        
        layout.addWidget(self.placeCellButton, *positions[0])
        layout.addWidget(self.hdCellButton, *positions[1])
        layout.addWidget(self.gridCellButton, *positions[2])
        layout.addWidget(self.boundaryCellButton, *positions[3])
        layout.addWidget(self.gradientCellButton, *positions[4])
        layout.addWidget(self.hdXPlaceCellButton, *positions[5])
        layout.addWidget(self.thetaCellButton, *positions[6])
        layout.addWidget(self.thetaSkipCellButton, *positions[7])  

        cellTypeBox = addGroupBox_2("Select Cell Type", "cellTypeBox")        
        cellTypeBox.setLayout(layout)
        
        self.cellTypeGroup = QtWidgets.QButtonGroup(cellTypeBox)
        self.cellTypeGroup.setObjectName(_fromUtf8("graphicFormatGroup"))
        self.cellTypeGroup.addButton(self.placeCellButton)
        self.cellTypeGroup.addButton(self.hdCellButton)
        self.cellTypeGroup.addButton(self.gridCellButton)
        self.cellTypeGroup.addButton(self.boundaryCellButton)
        self.cellTypeGroup.addButton(self.gradientCellButton)
        self.cellTypeGroup.addButton(self.hdXPlaceCellButton)
        self.cellTypeGroup.addButton(self.thetaCellButton)
        self.cellTypeGroup.addButton(self.thetaSkipCellButton)
        
        return cellTypeBox

    def selectAnalysisUi(self):        
       
        self.waveProperty = addCheckBox_2("Waveform Properties", "waveProperty")
        
        self.isi = addCheckBox_2("Interspike Interval", "isi")
        
        self.isiCorr = addCheckBox_2("ISI Autocorrelation", "isiCorr")
        
        self.thetaCell = addCheckBox_2("Theta-modulated Cell Index", "thetaCell")
        
        self.thetaSkipCell = addCheckBox_2("Theta-skipping Cell Index", "thetaSkipCell")        
        
        self.burst = addCheckBox_2("Burst Property", "burst")
        
        self.speed = addCheckBox_2("Spike Rate vs Running Speed", "speed")
        
        self.angVel = addCheckBox_2("Spike Rate vs Angular Velocity", "angVel")
        
        self.hdRate = addCheckBox_2("Spike Rate vs Head Direction", "hdRate")
        
        self.hdShuffle = addCheckBox_2("Head Directional Shuffling Analysis", "hdShuffle")
        
        self.hdTimeLapse = addCheckBox_2("Head Directional Time Lapse Analysis", "hdTimeLapse")
        
        self.hdTimeShift = addCheckBox_2("Head Directional Time Shift Analysis", "hdTimeShift")
        
        self.locRate = addCheckBox_2("Spike Rate vs Location", "locRate")
        
        self.locShuffle = addCheckBox_2("Locational Shuffling Analysiss", "locShuffle")
        
#        self.placeField = addCheckBox_2("Place Field Map", "placeField")
        
        self.locTimeLapse = addCheckBox_2("Locational Time Lapse Analysis", "locTimeLapse")
        
        self.locTimeShift = addCheckBox_2("Locational Time Shift Analysis", "locTimeShift")        

        self.spatialCorr = addCheckBox_2("Spatial Autocorrelation", "spatialCorr")                
        
        self.grid = addCheckBox_2("Grid Cell Analysis", "grid")
        
        self.border = addCheckBox_2("Border Cell Analysis", "border")
        
        self.gradient = addCheckBox_2("Gradient Cell Analysis", "gradient")
                
        self.mra = addCheckBox_2("Multiple Regression", "mra")
        
        self.interDepend = addCheckBox_2("Interdependence Analysis", "interDepend")
        
        self.lfpSpectrum = addCheckBox_2("LFP Frequency Spectrum", "lfpSpectrum")
        
        self.spikePhase = addCheckBox_2("Unit LFP-phase Distribution", "spikePhase")
        
        self.phaseLock = addCheckBox_2("Unit LFP-phase Locking", "phaseLock")
        
        self.lfpSpikeCausality = addCheckBox_2("Unit-LFP Causality", "lfpSpikeCausality")
        
        self.scrollLayout= QtWidgets.QVBoxLayout()
        self.scrollLayout.addWidget(self.waveProperty)
        self.scrollLayout.addWidget(self.isi)
        self.scrollLayout.addWidget(self.isiCorr)
        self.scrollLayout.addWidget(self.thetaCell)
        self.scrollLayout.addWidget(self.thetaSkipCell)
        self.scrollLayout.addWidget(self.burst)
        self.scrollLayout.addWidget(self.speed)
        self.scrollLayout.addWidget(self.angVel)
        self.scrollLayout.addWidget(self.hdRate)
        self.scrollLayout.addWidget(self.hdShuffle)
        self.scrollLayout.addWidget(self.hdTimeLapse)
        self.scrollLayout.addWidget(self.hdTimeShift)
        self.scrollLayout.addWidget(self.locRate)
        self.scrollLayout.addWidget(self.locShuffle)
#        self.scrollLayout.addWidget(self.placeField)
        self.scrollLayout.addWidget(self.locTimeLapse)
        self.scrollLayout.addWidget(self.locTimeShift)
        self.scrollLayout.addWidget(self.spatialCorr)
        self.scrollLayout.addWidget(self.grid)
        self.scrollLayout.addWidget(self.border)
        self.scrollLayout.addWidget(self.gradient)
        self.scrollLayout.addWidget(self.mra)
        self.scrollLayout.addWidget(self.interDepend)
        self.scrollLayout.addWidget(self.lfpSpectrum)
        self.scrollLayout.addWidget(self.spikePhase)
        self.scrollLayout.addWidget(self.phaseLock)
        self.scrollLayout.addWidget(self.lfpSpikeCausality)        
        
        self.functionWidget= ScrollableWidget()
        self.functionWidget.setContents(self.scrollLayout)
        
        self.funcSelectLabel = addLabel_2("Analysis Selection", "funcSelectLabel")
        self.selectAll = addCheckBox_2("Select All", "selectAll")

        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.funcSelectLabel, 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.selectAll, 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.functionWidget, 5)
        
        return layout
        
    def retranslateUi(self):
        self.setWindowTitle(_translate("MainWindow", "NeuroChaT", None))
        self.setWindowIcon(QtGui.QIcon("icon_48.png"))

    def start(self):
        self._getConfig()
        self._control.finished.connect(self.restoreStartButton)
        self._control.start()
#        self.nthread= QtCore.QThread()
#        self.worker= Worker(self.startAnalysis)
#        self.worker.moveToThread(self.nthread)
#        self.nthread.started.connect(self.worker.run)
#        [self.worker.finished.connect(x) for x in [self.restoreStartButton, self.nthread.quit]]
#        
#        self.nthread.start()

    def restoreStartButton(self):
        pdModel= PandasModel(self._control.getResults())        
        self._resultsUi.setData(pdModel)
        self._resultsUi.show()

    def export_results(self):
        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getSaveFileName(self, \
        'Export analysis results to...', os.getcwd()+ os.sep+ 'nc_results.xlsx', "Excel Files (*.xlsx .*xls)")[0])
        if not excel_file:
            logging.warning("No excel file selected! Results cannot be exported!")
        else:
            try:
                results= self._control.getResults()
                results.to_excel(excel_file)
                logging.info("Analysis results exported to: "+ \
                                            excel_file.rstrip("\n\r").split(os.sep)[-1])
            except:                                        
                logging.error('Failed to export results!')

    def export_graphicInfo(self):
        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getSaveFileName(self, \
        'Export information to...', os.getcwd()+ os.sep+ 'nc_graphicInfo.xlsx', "Excel Files (*.xlsx .*xls)")[0])
        if not excel_file:
            logging.warning("No excel file selected! Information cannot be exported!")
        else:
            try:
                info= self._control.getOutputFiles()
                info.to_excel(excel_file)
                logging.info("Graphics information exported to: "+ \
                                            excel_file.rstrip("\n\r").split(os.sep)[-1])
            except:                                        
                logging.error('Failed to export graphics information!')

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, "Message", \
            "Save current session before you quit?",\
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Close | QtWidgets.QMessageBox.Cancel,\
            QtWidgets.QMessageBox.Save)
        if reply== QtWidgets.QMessageBox.Save:
            config_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getSaveFileName(self, \
                        'Save configuration to...', os.getcwd()+ os.sep+ 'nc_config.ncfg', ".ncfg")[0])
            if config_file:
                try:
                    event.accept()
                except:
                    logging.error('Failed to save configuration!')
                    event.ignore()
        elif reply== QtWidgets.QMessageBox.Close:
            event.accept()
        else:
            event.ignore()
    def exit_nc(self):
        self.close()
#        print('ask for saving session...')
#        QtCore.QCoreApplication.instance().quit()
        
    def data_format_select(self, ind):
        data_format= self.fileFormatBox.itemText(ind)
        self._control.setDataFormat(data_format)
        logging.info("Input data format set to: " + data_format)
        self._setDictation()
        if data_format== 'Axona' or data_format== 'Neuralynx':
            self._control.setNwbFile('')
        
    def mode_select(self, ind):
        self._control.setAnalysisMode(ind)
        logging.info("Analysis mode set to: " + self.modeBox.itemText(ind))
        self._setDictation()
    def graphic_format_select(self):
        button= self.graphicFormatGroup.checkedButton()
        text= button.text()
        self._control.setGraphicFormat(text)
        logging.info("Graphic file format set to: " + text)
        
    def cell_type_select(self):
        button= self.cellTypeGroup.checkedButton()
        text= button.text()
        self._control.setCellType(text)
        logging.info("Cell type set to: " + text)
        self.cell_type_analysis(text)
        
    def select_all(self):
        if self.selectAll.isChecked()== True:
            logging.info("Selected ALL analyses")
            for checkbox in self.functionWidget.findChildren(QtWidgets.QCheckBox):
                checkbox.setChecked(True)
        else:
            logging.info("Deselected ALL analyses")
            for checkbox in self.functionWidget.findChildren(QtWidgets.QCheckBox):
                checkbox.setChecked(False)
                
    def lfpChan_getitems(self):
        file_format= self.fileFormatBox.itemText(self.fileFormatBox.currentIndex())
        items= [""]
        if file_format== "Neuralynx":
            files= os.listdir(os.getcwd())
            items= [f for f in files if f.endswith('ncs')]
            
        elif file_format== "Axona":
            files= os.listdir(os.getcwd())
            items= [f.split('.')[-1] for f in files if '.eeg' in f or '.egf' in f]
            
        elif file_format== "NWB":
            try:
                path = '/processing/Neural Continuous/LFP'
                hdf= Nhdf()
                hdf.setFilename(self._control.getNwbFile())
                if path in hdf.f:
                    items= list(hdf.f[path].keys())
                else:
                    logging.warning('No Lfp channel stored in the path:'+ path)
            except:
                logging.error('Cannot read the hdf file')
        else:
            items= [str(i) for i in list(range(256))]
        self.lfpChanBox.clear()
        self.lfpChanBox.addItems(items)
            
    def browse(self):
        mode_id= self.modeBox.currentIndex()
        file_format= self._control.getDataFormat()
        if mode_id==0 or mode_id==1:
            if file_format== "Axona" or file_format== "Neuralynx":
                if file_format== "Axona":
                    spike_filter= "".join(["*." + str(x)+ ";;" for x in list(range(1, 129))])
                    spatial_filter= "*.txt"
                elif file_format== "Neuralynx":
                    spike_filter= "*.ntt;; *.nst;; *.nse"
                    spatial_filter= "*.nvt"
                spike_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self, \
                               'Select spike file...', os.getcwd(), spike_filter)[0])
                if not spike_file:
                    logging.warning("No spike file selected")
                else:
                    words= spike_file.rstrip("\n\r").split(os.sep)
                    directory= os.sep.join(words[0:-1])
                    os.chdir(directory)
#                    spike_file= words[-1]
#                    self._currDir= directory
                    self._control.setSpikeFile(spike_file)
                    logging.info("New spike file added: " + \
                                        words[-1])
                    spatial_file= QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self, \
                               'Select spatial file...', os.getcwd(), spatial_filter)[0])
                    if not spatial_file:
                        logging.warning("No spatial file selected")
                    else:
                        words= spatial_file.rstrip("\n\r").split(os.sep)      
#                        spatial_file= words[-1]
                        self._control.setSpatialFile(spatial_file)
                        logging.info("New spatial file added: " + \
                                        words[-1])
                    self.lfpChan_getitems()
                    
            elif file_format== "NWB":
                nwb_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self, \
                               'Select NWB file...', os.getcwd(), "*.hdf5")[0])
                if not nwb_file:
                    logging.warning("No NWB file selected")
                else:
                    words= nwb_file.rstrip("\n\r").split(os.sep)
                    directory= os.sep.join(words[0:-1])
                    os.chdir(directory)
                    self._control.setNwbFile(nwb_file)
                    
                    logging.info("New NWB file added: "+ \
                    words[-1])
                    try:
                        hdf= Nhdf()
                        hdf.setFilename(nwb_file)
                        path = '/processing/Shank'
                        if path in hdf.f:
                            items= list(hdf.f[path].keys())
                        else:
                            logging.warning('No Shank data stored in the path:'+ path)
                        if items:
                            item, ok = QtWidgets.QInputDialog.getItem(self, "Select electrode group", 
                                      "Electrode groups: ", items, 0, False)
                            if ok:
                                self._control.setSpikeFile(nwb_file+ '+'+ path+ '/' + item)
                                logging.info('Spike data set to electrode group: '+ path+ '/'+ item)
                                
                                path = '/processing/Behavioural/Position'
                                if path in hdf.f:
                                    self._control.setSpatialFile(nwb_file+ '+'+ path)
                                    logging.info('Position data set to group: '+ path)
                                else:
                                    logging.warning(path+ ' not found! Spatial data cannot be set!')
                    except:
                        logging.error('Cannot read the hdf file')
            
                    self.lfpChan_getitems()
                    
        elif mode_id==2:
            excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self, \
                               'Select Excel file...', os.getcwd(), "*.xlsx;; .*xls")[0])
            if not excel_file:
                logging.warning("No excel file selected")
            else:
                words= excel_file.rstrip("\n\r").split(os.sep)
                directory= os.sep.join(words[0:-1])
                os.chdir(directory)
#                self._currDir= directory
##                excel_file= words[-1]
                self._control.setExcelFile(excel_file)
                logging.info("New excel file added: "+ \
                                        words[-1])
#        elif mode_id== 3:
#            data_directory = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getExistingDirectory(self, \
#                           'Select data directory...', os.getcwd()))
#            if not data_directory:
#                logging.warning("No data directory selected")
#            else:
##                self._currDir= data_directory
#                self._control.setDataDir(data_directory)
#                logging.info("New directory added: "+ data_directory)
    def update_log(self, msg):
        self.logText.insertLog(msg)
    def clear_log(self):
        self.logText.clear()
        logging.info("Log cleared!")
    def save_log(self):
#        self.logText.selectAll
        text= self.logText.getText()
        name = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getSaveFileName(self, 'Save log as...', os.getcwd(), "Text files(*.txt)")[0])
        if not name:
            logging.warning("File not specified. Log is not saved")
        else:
            try:
                file= open(name, 'w')            
                file.write(text)
                logging.info("Log saved in: "+ name)
            except:
                logging.error('Log is not saved! See if the file is open in another application!')
    def set_unit_no(self, value):
        self._control.setUnitNo(value)
        logging.info("Selected Unit: "+ str(value))
    
    def set_lfp_chan(self, value):
        lfpID= self.lfpChanBox.itemText(value)
        
        if lfpID:
            logging.info("Selected LFP channel: "+ lfpID)
            data_format= self._control.getDataFormat()
            if data_format == 'Axona':
                spikeFile = self._control.getSpikeFile()
                lfpFile= ''.join(spikeFile.split('.')[:-1])+ '.'+ lfpID
            elif data_format== 'Neuralynx':
                spikeFile = self._control.getSpikeFile()
                print(os.sep.join(spikeFile.split(os.sep)[:-1])+ os.sep+ lfpID)
                lfpFile= os.sep.join(spikeFile.split(os.sep)[:-1])+ os.sep+ lfpID
            elif data_format== 'NWB':
                nwbFile= self._control.getNwbFile()
                lfpFile= nwbFile+ '+' + '/processing/Neural Continuous/LFP'+ '/'+ lfpID
                # Will implement later
            else:
                logging.error('The input data format not supported!')
            self._control.setLfpFile(lfpFile)
        
    def _setDictation(self):
        _dictation= ["Select spike(.n) &/or position file(.txt)",
                     "Select spike(.n) &/or position file(.txt)",
                     "Select excel(.xls/.xlsx) file with unit list",
                     "Select folder"]                 
        file_format= self._control.getDataFormat()
        analysis_mode, mode_id= self._control.getAnalysisMode()
        if file_format== "Neuralynx":
            _dictation[:2]= ["Select spike(.ntt/.nst/.nse) &/or position file(.nvt)"]*2
        elif file_format== "NWB":
            _dictation[:2]= ["Select .hdf5 file"]*2
        self.filenameLine.setText(_dictation[mode_id])
    
    def _getConfig(self):
        #Get selected function from functioWidget
        for checkbox in self.functionWidget.findChildren(QtWidgets.QCheckBox):
            self._control.setAnalysis(checkbox.objectName(), checkbox.isChecked())      
            
        for spinbox in self._paramUi.findChildren(QtWidgets.QSpinBox):
            self._control.setParam(spinbox.objectName(), spinbox.value())
            
        for spinbox in self._paramUi.findChildren(QtWidgets.QDoubleSpinBox):
            self._control.setParam(spinbox.objectName(), spinbox.value())
            
        for combobox in self._paramUi.findChildren(QtWidgets.QComboBox):
            text= combobox.currentText()
            try:
                value= int(text)
                self._control.setParam(combobox.objectName(), value)
            except:
                self._control.setParam(combobox.objectName(), text)
                
    def save_session(self):
        ncfg_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getSaveFileName(self, 'Save session as...', os.getcwd(), "*.ncfg")[0])
        if not ncfg_file:
            logging.warning("File not specified. Session is not saved")
        else:
            self._getConfig()
            self._control.saveConfig(ncfg_file)
            logging.info("Session saved in: "+ ncfg_file)
            
    def load_session(self):
        ncfg_file= QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self, 'Select NCFG file...', os.getcwd(), "(*.ncfg)")[0])
        if not ncfg_file:
            logging.error("No saved session selected! Loading failed!")
        else:            
            self._control.loadConfig(ncfg_file)        

        index = self.fileFormatBox.findText(self._control.getDataFormat())
        if index >= 0:
            self.fileFormatBox.setCurrentIndex(index)
        mode, mode_id= self._control.getAnalysisMode()
        self.modeBox.setCurrentIndex(mode_id)
            
        getattr(self, self._control.getGraphicFormat()+ 'Button').setChecked(True)
        self.graphic_format_select()
        
        index= self._control.getUnitNo()
        if index>= 0 & index< 256:
            self.unitNoBox.setCurrentIndex(index)

        file= self._control.getLfpFile()
        if self._control.getDataFormat()== 'Axona':
            file_tag= file.split('.')[-1]
        elif self._control.getDataFormat()== 'Neuralynx':
            file_tag= file.split(os.sep)[-1].split('.')[0]
        elif self._control.getDataFormat()== 'NWB':
            file_tag= file.split('/')[-1]
            
        self.lfpChan_getitems()
        index=  self.lfpChanBox.findText(file_tag)
        if index >= 0:
            self.lfpChanBox.setCurrentIndex(index)
        
        cell_type= self._control.getCellType()
        select_by_type= False
        for button in self.cellTypeGroup.buttons():
            if button.text== cell_type:
                select_by_type= True
                button.click()
        if not select_by_type:
           for key in self._control.getAnalysisList():
               getattr(self, key).setChecked(self._control.getAnalysis(key))
                            
        paramList= self._control.getParamList()
        
        for name in paramList:
            paramWidget= getattr(self._paramUi, name)
            if isinstance(paramWidget, QtWidgets.QComboBox):
                index = paramWidget.findText(str(self._control.getParam(name)))
                if index>=0: 
                    paramWidget.setCurrentIndex(index)
            else:
                paramWidget.setValue(self._control.getParam(name))                
                    
    def merge_output(self):
        self._mergeUi.mergeEnable= True
        self._mergeUi.setWindowTitle(QtWidgets.QApplication.translate("mergeWindow", "Merge PDF/PS", None))
        self._mergeUi.show()
        logging.info("Tool to MERGE graphic files activated! Only PDF files can be merged!")
    def accumulate_output(self):
        self._mergeUi.mergeEnable= False
        self._mergeUi.setWindowTitle(QtWidgets.QApplication.translate("mergeWindow", "Accumulate PDF/PS", None))
        self._mergeUi.show()
        logging.info("Tool to ACCUMULATE graphic files activated")
    
    def compare_units(self):
        print('Compare')
        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self, \
        'Select unit-pair list...', os.getcwd(), "*.xlsx;; .*xls")[0])
        if not excel_file:
            logging.warning("No excel file selected! Comparing units is unsuccessful!")
        else:
            self._mergeUi.filenameLine.setText(excel_file)
            logging.info("New excel file added: "+ \
                                        excel_file.rstrip("\n\r").split(os.sep)[-1])
        excel_data= pd.read_excel(excel_file)
        print('Create an excel read warpper from PANDAS')
        print(excel_data)


    def verify_units(self):
        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self, \
        'Select data description list...', os.getcwd(), "*.xlsx;; .*xls")[0])
        if not excel_file:
            logging.warning("No excel file selected! Verification of units is unsuccessful!")
        else:
#            self._mergeUi.filenameLine.setText(excel_file)                    
            logging.info("New excel file added: "+ \
                                        excel_file.rstrip("\n\r").split(os.sep)[-1])            
            self._control.verify_units(excel_file)
    
    def convert_to_nwb(self):
        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self, \
        'Select data description list...', os.getcwd(), "*.xlsx;; .*xls")[0])
        if not excel_file:
            logging.warning("No excel file selected! Conversion to NWB is unsuccessful!")
        else:
#            self._mergeUi.filenameLine.setText(excel_file)                    
            logging.info("New excel file added: "+ \
                                        excel_file.rstrip("\n\r").split(os.sep)[-1])            
            self._control.convert_to_nwb(excel_file)
                    
    def set_parameters(self):
        self._paramUi.show()
                
    def cell_type_analysis(self, cell_type):
        if cell_type== "Place":
            self.waveProperty.setChecked(True)
            self.isi.setChecked(True)
            self.isiCorr.setChecked(True)

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.angVel.setChecked(True)
            
            self.hdRate.setChecked(False)
            self.hdShuffle.setChecked(False)
            self.hdTimeLapse.setChecked(False)
            self.hdTimeShift.setChecked(False)
            
            self.locRate.setChecked(True)
            self.locShuffle.setChecked(True)
#            self.placeField.setChecked(True)
            self.locTimeLapse.setChecked(True)
            self.locTimeShift.setChecked(True)

            self.spatialCorr.setChecked(True)
            
            
            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(False)
            
            self.mra.setChecked(True)
            self.interDepend.setChecked(True)
            
            self.thetaCell.setChecked(False)
            self.thetaSkipCell.setChecked(False)

            self.lfpSpectrum.setChecked(False)
            self.spikePhase.setChecked(False)
            self.phaseLock.setChecked(False)
            self.lfpSpikeCausality.setChecked(False)
            
        elif cell_type== "Head-directional":
            self.waveProperty.setChecked(True)
            self.isi.setChecked(True)
            self.isiCorr.setChecked(True)            

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.angVel.setChecked(True)
            
            self.hdRate.setChecked(True)
            self.hdShuffle.setChecked(True)
            self.hdTimeLapse.setChecked(True)
            self.hdTimeShift.setChecked(True)
            
            self.locRate.setChecked(False)
            self.locShuffle.setChecked(False)
#            self.placeField.setChecked(False)
            self.locTimeLapse.setChecked(False)
            self.locTimeShift.setChecked(False)
            
            self.spatialCorr.setChecked(False)
            
            
            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(False)
            
            self.mra.setChecked(True)
            self.interDepend.setChecked(True)
            
            self.thetaCell.setChecked(False)
            self.thetaSkipCell.setChecked(False)
            
            self.lfpSpectrum.setChecked(False)
            self.spikePhase.setChecked(False)
            self.phaseLock.setChecked(False)
            self.lfpSpikeCausality.setChecked(False)
            
        elif cell_type== "Grid":
            self.waveProperty.setChecked(True)
            self.isi.setChecked(True)
            self.isiCorr.setChecked(True)            

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.angVel.setChecked(True)
            
            self.hdRate.setChecked(False)
            self.hdShuffle.setChecked(False)
            self.hdTimeLapse.setChecked(False)
            self.hdTimeShift.setChecked(False)
            
            self.locRate.setChecked(True)            
            self.locShuffle.setChecked(False)            
#            self.placeField.setChecked(True)
            self.locTimeLapse.setChecked(True)            
            self.locTimeShift.setChecked(False)

            self.spatialCorr.setChecked(False)
            
            
            self.grid.setChecked(True)            
            self.border.setChecked(False)
            self.gradient.setChecked(False)
            
            self.mra.setChecked(True)
            self.interDepend.setChecked(True)
            
            self.thetaCell.setChecked(False)
            self.thetaSkipCell.setChecked(False)

            self.lfpSpectrum.setChecked(False)
            self.spikePhase.setChecked(False)
            self.phaseLock.setChecked(False)
            self.lfpSpikeCausality.setChecked(False)
            
        elif cell_type== "Boundary":
            self.waveProperty.setChecked(True)
            self.isi.setChecked(True)
            self.isiCorr.setChecked(True)

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.angVel.setChecked(True)
            
            self.hdRate.setChecked(False)
            self.hdShuffle.setChecked(False)
            self.hdTimeLapse.setChecked(False)
            self.hdTimeShift.setChecked(False)
            
            self.locRate.setChecked(True)
            self.locShuffle.setChecked(False)
#            self.placeField.setChecked(True)
            self.locTimeLapse.setChecked(True)            
            self.locTimeShift.setChecked(False)

            self.spatialCorr.setChecked(False)
            
            self.grid.setChecked(False)            
            self.border.setChecked(True)            
            self.gradient.setChecked(False)
            
            self.mra.setChecked(True)
            self.interDepend.setChecked(True)
            
            self.thetaCell.setChecked(False)
            self.thetaSkipCell.setChecked(False)

            self.lfpSpectrum.setChecked(False)
            self.spikePhase.setChecked(False)
            self.phaseLock.setChecked(False)
            self.lfpSpikeCausality.setChecked(False)
            
        elif cell_type== "Gradient":
            self.waveProperty.setChecked(True)
            self.isi.setChecked(True)
            self.isiCorr.setChecked(True)

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.angVel.setChecked(True)
            
            self.hdRate.setChecked(False)
            self.hdShuffle.setChecked(False)
            self.hdTimeLapse.setChecked(False)
            self.hdTimeShift.setChecked(False)
            
            self.locRate.setChecked(True)            
            self.locShuffle.setChecked(False)            
#            self.placeField.setChecked(True)
            self.locTimeLapse.setChecked(True)            
            self.locTimeShift.setChecked(False)

            self.spatialCorr.setChecked(False)
            
            self.grid.setChecked(False)            
            self.border.setChecked(False)            
            self.gradient.setChecked(True)
            
            self.mra.setChecked(True)
            self.interDepend.setChecked(True)
            
            self.thetaCell.setChecked(False)
            self.thetaSkipCell.setChecked(False)

            self.lfpSpectrum.setChecked(False)
            self.spikePhase.setChecked(False)
            self.phaseLock.setChecked(False)
            self.lfpSpikeCausality.setChecked(False)
            
        elif cell_type== "HDxPlace":
            self.waveProperty.setChecked(True)
            self.isi.setChecked(True)
            self.isiCorr.setChecked(True)            

            self.burst.setChecked(False)

            self.speed.setChecked(True)
            self.angVel.setChecked(True)
            
            self.hdRate.setChecked(True)
            self.hdShuffle.setChecked(True)
            self.hdTimeLapse.setChecked(True)
            self.hdTimeShift.setChecked(True)
            
            self.locRate.setChecked(True)
            self.locShuffle.setChecked(True)
#            self.placeField.setChecked(True)
            self.locTimeLapse.setChecked(True)
            self.locTimeShift.setChecked(True)
            
            self.spatialCorr.setChecked(True)
            
            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(False)
            
            self.mra.setChecked(True)
            self.interDepend.setChecked(True)
            
            self.thetaCell.setChecked(False)
            self.thetaSkipCell.setChecked(False)
            
            self.lfpSpectrum.setChecked(False)
            self.spikePhase.setChecked(False)
            self.phaseLock.setChecked(False)
            self.lfpSpikeCausality.setChecked(False)
            
        elif cell_type== "Theta-rhythmic":
            self.waveProperty.setChecked(True)
            self.isi.setChecked(True)
            self.isiCorr.setChecked(True)            

            self.burst.setChecked(True)

            self.speed.setChecked(True)
            self.angVel.setChecked(True)
            
            self.hdRate.setChecked(False)
            self.hdShuffle.setChecked(False)
            self.hdTimeLapse.setChecked(False)
            self.hdTimeShift.setChecked(False)
            
            self.locRate.setChecked(False)
            self.locShuffle.setChecked(False)
#            self.placeField.setChecked(False)
            self.locTimeLapse.setChecked(False)
            self.locTimeShift.setChecked(False)
            
            self.spatialCorr.setChecked(False)
            
            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(False)
            
            self.mra.setChecked(False)
            self.interDepend.setChecked(False)
            
            self.thetaCell.setChecked(True)
            self.thetaSkipCell.setChecked(False)
            
            self.lfpSpectrum.setChecked(True)
            self.spikePhase.setChecked(True)
            self.phaseLock.setChecked(True)
            self.lfpSpikeCausality.setChecked(True)
            
        elif cell_type== "Theta-skipping":
            self.waveProperty.setChecked(True)
            self.isi.setChecked(True)
            self.isiCorr.setChecked(True)            

            self.burst.setChecked(True)

            self.speed.setChecked(True)
            self.angVel.setChecked(True)
            
            self.hdRate.setChecked(False)
            self.hdShuffle.setChecked(False)
            self.hdTimeLapse.setChecked(False)
            self.hdTimeShift.setChecked(False)
            
            self.locRate.setChecked(False)
            self.locShuffle.setChecked(False)
#            self.placeField.setChecked(False)
            self.locTimeLapse.setChecked(False)
            self.locTimeShift.setChecked(False)
            
            self.spatialCorr.setChecked(False)
            
            
            self.grid.setChecked(False)
            self.border.setChecked(False)
            self.gradient.setChecked(False)
            
            self.mra.setChecked(False)
            self.interDepend.setChecked(False)
            
            self.thetaCell.setChecked(False)            
            self.thetaSkipCell.setChecked(True)
            
            self.lfpSpectrum.setChecked(True)
            self.spikePhase.setChecked(True)
            self.phaseLock.setChecked(True)
            self.lfpSpikeCausality.setChecked(True)        


class Ui_results(QtWidgets.QDialog):
    def __init__(self, parent= None):
        super().__init__(parent)
    
    def setupUi(self):
        self.setObjectName(_fromUtf8("resultsWindow"))
        self.setEnabled(True)
        self.setFixedSize(725, 220)
        self.setWindowTitle(QtWidgets.QApplication.translate("resultsWindow", "Analysis results", None))
                
        # layout
        self.layout = QtWidgets.QVBoxLayout()
        
        self.table = QtWidgets.QTableView()
        self.table.resizeColumnsToContents()
        self.table.showGrid()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        
        self.exportButton = addPushButton(self, (30, 50, 50, 23), "exportResult", "Export")
        
        self.layout.addWidget(self.exportButton)
        self.layout.addWidget(self.table)        
        self.setLayout(self.layout)
        self.setDefault()
        
    def setData(self, pdModel):
        self.table.setModel(pdModel)
#        self.show()

    def setDefault(self):
        pass
        
class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a QT table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, index, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[index]
        elif orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[index]
        return None
    
class Ui_merge(QtWidgets.QDialog):
    def __init__(self, parent= None):
        super().__init__(parent)
        self.mergeEnable= True
        self._getFilesUi= Ui_getfiles(self)
        
        self.setupUi()
        self.behaviourUi()   

        self.files= []
        self.dst_directory= [] # destination directory for accumulation
        self.dst_file= [] # destination files for mergeing
        
    def setupUi(self):
        self.setObjectName(_fromUtf8("mergeWindow"))
        self.setEnabled(True)
        self.setFixedSize(324, 220)
        
        self.useListButton = addRadioButton(self, (10, 20, 102, 23), "useList", "Use Excel List")

        self.chooseFilesButton = addRadioButton(self, (10, 80, 102, 23), "chooseFiles", "Choose files")
        
        self.inputFormatGroup = QtWidgets.QButtonGroup(self)
        self.inputFormatGroup.setObjectName(_fromUtf8("mergeButtonGroup"))
        self.inputFormatGroup.addButton(self.useListButton)
        self.inputFormatGroup.addButton(self.chooseFilesButton) 
        
        self.browseExcelButton = addPushButton(self, (30, 50, 50, 23), "browseExcel", "Browse")
        
        self.filenameLine = addLineEdit(self, (95, 50, 215, 23), "filename", "")
        self.filenameLine.setText("Select Excel (.xls/.xlsx) file")
        
        self.selectButton = addPushButton(self, (30, 110, 70, 23), "select", "Select now")
        
        self.saveInButton = addPushButton(self, (30, 140, 50, 23), "saveIn", "Save In")
        
        self.saveFilenameLine = addLineEdit(self, (95, 140, 215, 23), "saveFilename", "")
        
        self.startButton = addPushButton(self, (95, 180, 40, 23), "start", "Start")
        
        self.cancelButton = addPushButton(self, (165, 180, 50, 23), "cancel", "Cancel")
        
        self.setDefault()
        
        self._getFilesUi.setupUi()
        
    def setDefault(self):
        self.useListButton.setChecked(True)
        self.selectButton.setEnabled(False)
        
    def behaviourUi(self):
        self._getFilesUi.behaviourUi()        
        
        self.inputFormatGroup.buttonClicked.connect(self.merge_files)
        self.browseExcelButton.clicked.connect(self.browse_excel_merge)
        self.selectButton.clicked.connect(self.select_files_merge)
        self.saveInButton.clicked.connect(self.save_in_merge)
        self.startButton.clicked.connect(self.start)
        self.cancelButton.clicked.connect(self.close)
        
    def merge_files(self):
        button= self.inputFormatGroup.checkedButton()
        if button.objectName()== "useList":
            self.browseExcelButton.setEnabled(True)
            self.filenameLine.setEnabled(True)
            self.selectButton.setEnabled(False)
            logging.info("Browse an excel list of output graphic files")
        elif button.objectName()== "chooseFiles":
            self.browseExcelButton.setEnabled(False)
            self.filenameLine.setEnabled(False)
            self.selectButton.setEnabled(True)
            logging.info("Select output graphic files")
    def browse_excel_merge(self):
        self.files= []        
        excel_file = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(self, \
        'Select output graphic file list...', os.getcwd(), "*.xlsx;; .*xls")[0])
        if not excel_file:
            logging.warning("No excel file selected! Merging/accumulating is unsuccessful!")
        else:
            self.filenameLine.setText(excel_file)
            logging.info("New excel file added: "+ \
                                        excel_file.rstrip("\n\r").split(os.sep)[-1])
        data= pd.read_excel(excel_file)
        self.files= data.values.T.tolist()[0]
        for i, f in enumerate(self.files):
            if not os.path.exists(f):
                self.files.pop(i)
                logging.warning(f+ ' file does not exist!')  
                
    def select_files_merge(self):
        self._getFilesUi.show()
        self.files= self._getFilesUi.get_files()

    def save_in_merge(self):
        if self.mergeEnable:
            self.dst_file= QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getSaveFileName(self, 'Save as...', os.getcwd(), "*.pdf;; *.ps")[0])            
            self.saveFilenameLine.setText(self.dst_file)
            if os.path.exists(self.dst_file):
                logging.info('PDF files will be merged to '+ self.dst_file)
        else:
            self.dst_directory= QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getExistingDirectory(self, \
                           'Select data directory...', os.getcwd()))
            self.saveFilenameLine.setText(self.dst_directory)
            if os.path.exists(self.dst_directory):
                logging.info('Files will be accumulated to '+ self.dst_directory)
    
    def start(self):
        if self.files:
            if self.mergeEnable:
                merger= PdfFileMerger()
                for f in self.files:
                    if os.path.exists(f):
                        merger.append(PdfFileReader(f, 'rb'))
                    else:
                        logging.warning('Cannot merge file '+ f)
            try:
                merger.write(self.dst_file)
                logging.info('Files merged to '+ self.dst_file)
            except:
                logging.error('Cannot merge files to '+ self.dst_directory)
            else:
                if os.path.exists(self.dst_directory) and os.access(self.dst_directory, os.W_OK):
                   for f in self.files:
                       try:
                           shutil.move(f, os.path.join(self.dst_directory, f.split(os.sep)[-1]))
                       except:
                           logging.warning('Cannot move file ' + f+ ' to '+ self.dst_directory)
                   logging.info('Files moved to '+ self.dst_directory)
                else:
                    logging.error('Destination folder '+ self.dst_directory+ ' does not exist or not write accessible!')        
        
class Ui_convert(QtWidgets.QDialog):
    def __init__(self, parent= None):
        super().__init__(parent)
    def setupUi(self):
        
        self.setObjectName(_fromUtf8("convertWindow"))
        self.setEnabled(True)
        self.setFixedSize(324, 220)
        self.setWindowTitle("Convert Files")
        
        self.inpFormatLabel = addLabel(self, (20, 10, 111, 17), "inpFormatLabel", "Convert From")
        
        self.fileFormatBox = addComboBox(self, (20, 30, 111, 22), "fileFormatBox")
        self.fileFormatBox.addItems(["Axona", "Neuralynx"])
        
        self.browseExcelButton = addPushButton(self, (30, 60, 60, 23), "browseExcel", "Browse")
        
        self.filenameLine = addLineEdit(self, (95, 60, 215, 23), "filename", "")
        self.filenameLine.setText("Select Excel (.xls/.xlsx) file")
                    
        
class Ui_parameters(QtWidgets.QDialog):
    def __init__(self, parent= None):
        super().__init__(parent)
        self.parent= parent
        self.setupUi()
        self.behaviourUi()
    def setupUi(self):
        self.setObjectName(_fromUtf8("paramSetWindow"))
        self.setEnabled(True)
        self.setFixedSize(640, 320)
        self.setWindowTitle(QtWidgets.QApplication.translate("paramSetWindow", "Parameter settings", None))    
        
        self.selectLabel = addLabel(self, (10, 10, 180, 17), "inpFormatLabel", "Select analysis to set parameters")
        
        self.paramList= QtWidgets.QListWidget(self)
        self.paramList.setGeometry(QtCore.QRect(10, 30, 170, 280))
        self.paramList.setObjectName(_fromUtf8("paramList"))
        
        items= []
        widgetNames= []
        for checkbox in self.parent.functionWidget.findChildren(QtWidgets.QCheckBox):
            items.append(checkbox.text())
            widgetNames.append(checkbox.objectName())            
        
        self.paramList.addItems(items)
       
        self.paramStack= QtWidgets.QStackedWidget(self)
        self.paramStack.setGeometry(QtCore.QRect(190, 30, 440, 280))
        self.paramStack.setObjectName(_fromUtf8("paramStack"))
    
        self.paramStack.addWidget(self.waveformPage())
        
        self.paramStack.addWidget(self.isiPage())
        
        self.paramStack.addWidget(self.isiCorrPage())
        
        self.paramStack.addWidget(self.thetaCellPage())
        
        self.paramStack.addWidget(self.thetaSkipCellPage())
        
        self.paramStack.addWidget(self.burstPage())
        
        self.paramStack.addWidget(self.speedPage())

        self.paramStack.addWidget(self.angVelPage())

        self.paramStack.addWidget(self.hdRatePage())
        
        self.paramStack.addWidget(self.hdShufflePage())

        self.paramStack.addWidget(self.hdTimeLapsePage())
                
        self.paramStack.addWidget(self.hdTimeShiftPage())
                
        self.paramStack.addWidget(self.locRatePage())
        
        self.paramStack.addWidget(self.locShufflePage())
        
#        self.paramStack.addWidget(self.placeFieldPage())
        
        self.paramStack.addWidget(self.locTimeLapsePage())
        
        self.paramStack.addWidget(self.locTimeShiftPage())
        
        self.paramStack.addWidget(self.spatialCorrPage())        
        
        self.paramStack.addWidget(self.gridPage())
        
        self.paramStack.addWidget(self.borderPage())
        
        self.paramStack.addWidget(self.gradientPage())

        self.paramStack.addWidget(self.mraPage())
        
        self.paramStack.addWidget(self.interDependPage())
        
        self.paramStack.addWidget(self.lfpSpectrumPage())
        
        self.paramStack.addWidget(self.spikePhasePage())
        
        self.paramStack.addWidget(self.phaseLockPage())
        
        self.paramStack.addWidget(self.lfpSpikeCausalityPage())
    
    def behaviourUi(self):
        self.paramList.itemActivated.connect(self.change_stack_page)
        self.locRateFilter.activated[str].connect(self.set_loc_rate_filter)
        self.spatialCorrFilter.activated[str].connect(self.set_spat_corr_filter)
        self.paramList.itemActivated.connect(self.change_stack_page)
        
    def change_stack_page(self):
        self.paramStack.setCurrentWidget(self.paramStack.widget(self.paramList.currentRow()))
    
    def set_loc_rate_filter(self, filt_type):
        if filt_type== "Gaussian" :
            self.locRateKernLen.setSingleStep(1)
            self.locRateKernLen.setValue(3)
        elif filt_type== "Box" :
            self.locRateKernLen.setSingleStep(2)
            self.locRateKernLen.setValue(5)
            
    def set_spat_corr_filter(self, filt_type):
        if filt_type== "Gaussian" :
            self.spatialCorrKernLen.setSingleStep(1)
            self.spatialCorrKernLen.setValue(3)
        elif filt_type== "Box" :
            self.spatialCorrKernLen.setSingleStep(2)
            self.spatialCorrKernLen.setValue(5)
        
    def waveformPage(self):
        widget= ScrollableWidget()
        self.waveform_gb1= addGroupBox_2("", "waveform_gb1")
        
        boxLayout= QtWidgets.QVBoxLayout()
        boxLayout.addWidget(QtWidgets.QLabel("No parameter to set"))
        self.waveform_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.waveform_gb1)
        
        widget.setContents(layout)
        return widget
        
    def isiPage(self):
        widget= ScrollableWidget()
        # Box- 1
        self.isi_gb1= addGroupBox_2("Histogram", "isi_gb1")
        
        self.isiBin= addSpinBox_2(1, 50, "isiBin")
        self.isiBin.setValue(2)
        
        self.isiLength= addSpinBox_2(10, 1000, "isiLength")
        self.isiLength.setValue(350)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Histogram Binsize", self.isiBin, "ms [range: 1-50]")
        boxLayout.addRow( "Histogram Length", self.isiLength, "ms [range: 10-1000]")
        
        self.isi_gb1.setLayout(boxLayout)
        
        # Box- 2
        self.isi_gb2= addGroupBox_2("log-log plot", "isi_gb2") 
        
        self.isiLogNoBins= addSpinBox_2(10, 100, "isiLogNoBins")
        self.isiLogNoBins.setValue(70)
        
        self.isiLogLength= addSpinBox_2(10, 1000, "isiLogLength")
        self.isiLogLength.setValue(350)
                
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("No. of Histogram Bins", self.isiLogNoBins, "[range: 10-100]")
        boxLayout.addRow("Histogram Length", self.isiLogLength, "ms [range: 10-1000]")

        self.isi_gb2.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.isi_gb1)
        layout.addWidget(self.isi_gb2)
        
        widget.setContents(layout)
        
        return widget
        
        
    def isiCorrPage(self):
        widget= ScrollableWidget()
        
        # Box- 1
        self.isiCorr_gb1= addGroupBox_2("Zoomed In", "isiCorr_gb1")
        
        self.isiCorrBinShort= addSpinBox_2(1, 10, "isiCorrBinShort")
        self.isiCorrBinShort.setValue(1)
        
        self.isiCorrLenShort= addSpinBox_2(5, 50, "isiCorrLenShort")
        self.isiCorrLenShort.setValue(10)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Autcorrelation Histogram Binsize", self.isiCorrBinShort, "ms [range: 1-10]")
        boxLayout.addRow("Autocorrelation Length", self.isiCorrLenShort, "ms [range: 5-50]")
        
        self.isiCorr_gb1.setLayout(boxLayout)
        
        # Box- 2        
        self.isiCorr_gb2= addGroupBox_2("Zoomed Out", "isiCorr_gb2")      
        
        self.isiCorrBinLong= addSpinBox_2(1, 50, "isiCorrBinLong")
        self.isiCorrBinLong.setValue(2)
        
        self.isiCorrLenLong= addSpinBox_2(10, 1000, "isiCorrLenLong")
        self.isiCorrLenLong.setValue(350)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Autcorrelation Histogram Binsize", self.isiCorrBinLong, "ms [range: 1-50]")
        boxLayout.addRow("Autocorrelation Length", self.isiCorrLenLong, "ms [range: 10-1000]")
        
        self.isiCorr_gb2.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.isiCorr_gb1)
        layout.addWidget(self.isiCorr_gb2)
        
        widget.setContents(layout)
        return widget

    def thetaCellPage(self):
        widget= ScrollableWidget()
        self.thetaCell_gb1= addGroupBox_2("Curve Fitting Parameters", "thetaCell_gb1")

        self.thetaCellFreqMin= addDoubleSpinBox_2(1, 10, "thetaCellFreqMin")
        self.thetaCellFreqMin.setValue(6)
        self.thetaCellFreqMin.setSingleStep(0.5)
        
        self.thetaCellFreqMax= addDoubleSpinBox_2(8, 16, "thetaCellFreqMax")
        self.thetaCellFreqMax.setValue(12)
        self.thetaCellFreqMax.setSingleStep(0.5)
        
        self.thetaCellFreqStart= addDoubleSpinBox_2(5, 10, "thetaCellFreqStart")
        self.thetaCellFreqStart.setValue(6)
        self.thetaCellFreqStart.setSingleStep(0.5)
        
        self.thetaCellTau1Max= addDoubleSpinBox_2(0.5, 15, "thetaCellTau1Max")
        self.thetaCellTau1Max.setValue(5)
        self.thetaCellTau1Max.setSingleStep(0.5)
        
        self.thetaCellTau1Start= addDoubleSpinBox_2(0, 15, "thetaCellTau1Start")
        self.thetaCellTau1Start.setValue(0.1)
        self.thetaCellTau1Start.setSingleStep(0.05)
        
        self.thetaCellTau2Max= addDoubleSpinBox_2(0, 0.1, "thetaCellTau2Max")
        self.thetaCellTau2Max.setValue(0.05)
        self.thetaCellTau2Max.setSingleStep(0.005)
        
        self.thetaCellTau2Start= addDoubleSpinBox_2(0, 0.1, "thetaCellTau2Start")
        self.thetaCellTau2Start.setValue(0.05)
        self.thetaCellTau2Start.setSingleStep(0.005)     
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Minimum Frequency", self.thetaCellFreqMin, "Hz [range: 1-10, step: 0.5]")
        boxLayout.addRow("Maximum Frequency", self.thetaCellFreqMax, "Hz [range: 8-16, step: 0.5]")
        boxLayout.addRow("Starting Frequency", self.thetaCellFreqStart, "Hz [range: 5-10, step: 0.5]")
        boxLayout.addRow("Max Time Constant (Tau-1)", self.thetaCellTau1Max, "sec [range: 0.5-10, step: 0.5]") 
        boxLayout.addRow("Starting Tau-1", self.thetaCellTau1Start, "sec [range: 0-15, step: 0.05]")
        boxLayout.addRow("Gaussian Time Constant (Tau-2)", self.thetaCellTau2Max, "sec [range: 0-0.1, step: 0.005]") 
        boxLayout.addRow("Start of Tau-2", self.thetaCellTau2Start, "sec [range: 0-0.1, step: 0.005]")
        
        self.thetaCell_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.thetaCell_gb1)
        
        widget.setContents(layout)
        
        return widget
        
    def thetaSkipCellPage(self):
        widget= ScrollableWidget()
        self.thetaSkipCell_gb1= addGroupBox_2("Curve Fitting Parameters", "thetaSkipCell_gb1")
        
        boxLayout= QtWidgets.QVBoxLayout()
        boxLayout.addWidget(QtWidgets.QLabel("Uses the parameters from 'Theta-modulated Cell Index' analysis\n"+ \
        "The fitting parameters for 2nd frequency component is derived \n\rfrom the 1st component"))
        self.thetaSkipCell_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.thetaSkipCell_gb1)
        
        widget.setContents(layout)
        
        return widget
        
    def burstPage(self):
        widget= ScrollableWidget()
        
        # Box- 1
        self.burst_gb1= addGroupBox_2("Bursting conditions", "burst_gb1")
        
        self.burstThresh= addSpinBox_2(1, 15, "burstThresh")
        self.burstThresh.setValue(5)
        
        self.spikesToBurst= addSpinBox_2(2, 10, "spikesToBurst")
        self.spikesToBurst.setValue(2)
  
        self. ibiThresh= addSpinBox_2(5, 1000, "ibiThresh")
        self.ibiThresh.setValue(50)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Burst Threshold", self.burstThresh, "ms [range: 1-15]")
        boxLayout.addRow("Spikes to Burst", self.spikesToBurst, "[range: 2-10]")
        boxLayout.addRow("Interburst Interval Lower Cutoff", self.ibiThresh, "ms [range: 5-1000]")        
        
        self.burst_gb1.setLayout(boxLayout)
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.burst_gb1)
        
        widget.setContents(layout)
        return widget
        
    def speedPage(self):

        widget= ScrollableWidget()
        self.speed_gb1= addGroupBox_2("Analyses Parameters", "speed_gb1")
                
        self.speedBin= addSpinBox_2(1, 10, "speedBin")
        self.speedBin.setValue(1)
        
        self.speedMin= addSpinBox_2(0, 10, "speedMin")
        self.speedMin.setValue(0)
                
        self.speedMax= addSpinBox_2(10, 200, "speedMax")
        self.speedMax.setValue(40)
        
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Speed Binsize", self.speedBin, "cm/sec [range: 1-10]")
        boxLayout.addRow("Minimum Speed", self.speedMin, "cm/sec [range: 0-10]")
        boxLayout.addRow("Maximum Speed", self.speedMax, "cm/sec [range: 10-200]")
        
        self.speed_gb1.setLayout(boxLayout)
        
        self.speed_gb2= addGroupBox_2("Smoothing Box Kernal Length", "speed_gb2")
                
        self.speedKernLen= addSpinBox_2(1, 25, "speedKernLen")
        self.speedKernLen.setValue(3)
        self.speedKernLen.setSingleStep(2)
        # set validator in controller to accept only the odd numbers        
        
        self.speedRateKernLen= addSpinBox_2(1, 7, "speedRateKernLen")
        self.speedRateKernLen.setValue(3)
        self.speedRateKernLen.setSingleStep(2)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Speed", self.speedKernLen, "samples [range: 1-25, odds]")
        boxLayout.addRow("Spike Rate", self.speedRateKernLen, "bins [range: 1-7, odds]")
        
        self.speed_gb2.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.speed_gb1)
        layout.addWidget(self.speed_gb2)
        
        widget.setContents(layout)
        return widget
        
        
    def angVelPage(self):
        
        widget= ScrollableWidget()
        
        self.angVel_gb1= addGroupBox_2("Analyses Parameters", "angVel_gb1")
        
        self.angVelBin= addSpinBox_2(5, 50, "angVelBin")
        self.angVelBin.setValue(10)
        self.angVelBin.setSingleStep(5)

        self.angVelMin= addSpinBox_2(-500, 0, "angVelMin")
        self.angVelMin.setValue(-200)
        
        self.angVelMax= addSpinBox_2(0, 500, "angVelMax")
        self.angVelMax.setValue(200)
        
        self.angVelCutoff= addSpinBox_2(0, 100, "angVelCutoff")
        self.angVelCutoff.setValue(10)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Angular Velocity Binsize", self.angVelBin, "deg/sec [range: 1-50]")
        boxLayout.addRow("Minimum Velocity", self.angVelMin, "deg/sec [range: -500 to 0]") 
        boxLayout.addRow("Maximum Velocity", self.angVelMax, "deg/sec [range: 0 to 500]")
        boxLayout.addRow("Cutoff Velocity", self.angVelCutoff, "deg/sec [range: 0 to 100]")
        
        self.angVel_gb1.setLayout(boxLayout)
        
        self.angVel_gb2= addGroupBox_2("Smoothing Box Kernal Length", "angVel_gb2")
                
        self.angVelKernLen= addSpinBox_2(1, 25, "angVelKernLen")
        self.angVelKernLen.setValue(3)
        self.angVelKernLen.setSingleStep(2)
        # Set controller to recieve odd values
                
        self.angVelRateKernLen= addSpinBox_2(1, 5, "angVelRateKernLen")
        self.angVelRateKernLen.setValue(3)
        self.angVelRateKernLen.setSingleStep(2)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Head Direction", self.angVelKernLen, "samples [range: 1-25, odds]")
        boxLayout.addRow("Spike Rate", self.angVelRateKernLen, "bins [range: 1-5, odds]") 
        
        self.angVel_gb2.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.angVel_gb1)
        layout.addWidget(self.angVel_gb2)
        
        widget.setContents(layout)
        return widget
        
    def hdRatePage(self):
        widget= ScrollableWidget()
        self.hdRate_gb1= addGroupBox_2("Analyses Paramters", "hdRate_gb1")        

        self.hdBin= addComboBox_2("hdBin")
        hdBinItems= [str(d) for d in range(1, 360) if 360 % d== 0 and d>=5 and d<=45 ]
        self.hdBin.addItems(hdBinItems)

        self.hdAngVelCutoff= addSpinBox_2(0, 100, "hdAngVelCutoff")
        self.hdAngVelCutoff.setValue(10)
        self.hdAngVelCutoff.setSingleStep(5)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Head Directional Binsize", self.hdBin, "degree")
        boxLayout.addRow("Angular Velocity Cutoff", self.hdAngVelCutoff, "deg/sec [range: 0-100, step: 5]")
        
        self.hdRate_gb1.setLayout(boxLayout)
        
        self.hdRate_gb2= addGroupBox_2("Smoothing Box Kernal Length", "hdRate_gb2")
        
        self.hdRateKernLen= addSpinBox_2(1, 11, "hdRateKernLen")
        self.hdRateKernLen.setValue(5)
        self.hdRateKernLen.setSingleStep(2)
                
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Spike Rate", self.hdRateKernLen, "bins [range: 1-11, odds]")
        
        self.hdRate_gb2.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.hdRate_gb1)
        layout.addWidget(self.hdRate_gb2)
        
        widget.setContents(layout)        
        
        return widget
        
    def hdShufflePage(self):
        widget= ScrollableWidget()
        self.hdShuffle_gb1= addGroupBox_2("Analyses Paramters", "hdShuffle_gb1")        

        self.hdShuffleTotal= addSpinBox_2(100, 10000, "hdShuffleTotal")
        self.hdShuffleTotal.setValue(500)
        self.hdShuffleTotal.setSingleStep(50)
        
        self.hdShuffleLimit= addSpinBox_2(0, 500, "hdShuffleLimit")
        self.hdShuffleLimit.setValue(0)
        self.hdShuffleLimit.setSingleStep(2)
        
        self.hdShuffleNoBins= addSpinBox_2(10, 200, "hdShuffleNoBins")
        self.hdShuffleNoBins.setValue(100)
        self.hdShuffleNoBins.setSingleStep(10)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("No of Shuffles", self.hdShuffleTotal, "[range: 100-10000, step: 50]")
        boxLayout.addRow("Shuffling Limit", self.hdShuffleLimit, "sec [range: 0-500, 0 for Random, step: 2]")
        boxLayout.addRow("No of Histogram Bins", self.hdShuffleNoBins, "[range: 10-200, step: 10]") 
        
        self.hdShuffle_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.hdShuffle_gb1)
        
        widget.setContents(layout)
        
        return widget
        
    def hdTimeLapsePage(self):
        widget= ScrollableWidget()

        self.hdTimeLapse_gb1= addGroupBox_2("", "hdTimeLapse_gb1")
        
        boxLayout= QtWidgets.QVBoxLayout()
        boxLayout.addWidget(QtWidgets.QLabel("No parameter to set"))
        self.hdTimeLapse_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.hdTimeLapse_gb1)
        
        widget.setContents(layout)
        return widget
        
    def hdTimeShiftPage(self):
        widget= ScrollableWidget()
        
        self.hdTimeShift_gb1= addGroupBox_2("Shift Specifications", "hdTimeShift_gb1")
        
        self.hdShiftMax= addSpinBox_2(1, 100, "hdShiftMax")
        self.hdShiftMax.setValue(10)
        
        self.hdShiftMin= addSpinBox_2(-100, -1, "hdShiftMin")
        self.hdShiftMin.setValue(-10)      
        
        self.hdShiftStep= addSpinBox_2(1, 3, "hdShiftStep")
        self.hdShiftStep.setValue(1)

        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Maximum Shift", self.hdShiftMax, "indices [range: 1 to 100]")
        boxLayout.addRow("Minimum Shift", self.hdShiftMin, "indices [range: -1 to -100]")
        boxLayout.addRow("Index Steps", self.hdShiftStep, "[1, 2, 3]")
        
        self.hdTimeShift_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.hdTimeShift_gb1)
        
        widget.setContents(layout)
        return widget
        
    def locRatePage(self):
        widget= ScrollableWidget()
        # Box- 1
        self.locRate_gb1= addGroupBox_2("Analyses Paramters", "locRate_gb1")        
        
        self.locPixelSize= addSpinBox_2(1, 100, "locPixelSize")
        self.locPixelSize.setValue(3)

        self.locChopBound= addSpinBox_2(3, 20, "locChopBound")
        self.locChopBound.setValue(5)

#        self.locAngVelCutoff= addSpinBox_2(0, 100, "locAngVelCutoff")
#        self.locAngVelCutoff.setValue(30)
#        self.locAngVelCutoff.setSingleStep(5)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Pixel Size", self.locPixelSize, "cm [range: 1-100]")
        boxLayout.addRow("Bound for Chopping Edges", self.locChopBound, "pixels [range: 3-20]")
#        boxLayout.addRow("Angular Velocity Cutoff", self.locAngVelCutoff, "deg/sec [range: 0-100, step: 5]")
        
        self.locRate_gb1.setLayout(boxLayout)
        
#        # Box- 2
        self.locRate_gb2= addGroupBox_2("Smoothing Box Kernal", "locRate_gb2")
        
        self.locRateFilter= addComboBox_2("locRateFilter")
        self.locRateFilter.addItems(["Box", "Gaussian"])
        
        self.locRateKernLen= addSpinBox_2(1, 11, "locRateKernLen")
        self.locRateKernLen.setValue(5)
        self.locRateKernLen.setSingleStep(2)
        # Change step size to 0.5 if Gaussian is selected
                
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Smoothing Filter", self.locRateFilter, "")
        boxLayout.addRow("Spike Rate Pixels/Sigma", self.locRateKernLen, \
        "[range: 1-11]\n\r Box: odds")
        
        self.locRate_gb2.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.locRate_gb1)
        layout.addWidget(self.locRate_gb2)
        
        widget.setContents(layout)        
        
        return widget
        
    def locShufflePage(self):
        widget= ScrollableWidget()
        self.locShuffle_gb1= addGroupBox_2("Analyses Parameters", "locShuffle_gb1")        

        self.locShuffleTotal= addSpinBox_2(100, 10000, "locShuffleTotal")
        self.locShuffleTotal.setValue(500)
        self.locShuffleTotal.setSingleStep(50)
        
        self.locShuffleLimit= addSpinBox_2(0, 500, "locShuffleLimit")
        self.locShuffleLimit.setValue(0)
        self.locShuffleLimit.setSingleStep(2)
        
        self.locShuffleNoBins= addSpinBox_2(10, 200, "locShuffleNoBins")
        self.locShuffleNoBins.setValue(100)
        self.locShuffleNoBins.setSingleStep(10)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("No of Shuffles", self.locShuffleTotal, "[range: 100-10000, step: 50]")
        boxLayout.addRow("Shuffling Limit", self.locShuffleLimit, "sec [range: 0-500, 0 for Random, step: 2]")
        boxLayout.addRow("No of Histogram Bins", self.locShuffleNoBins, "[range: 10-200, step: 10]") 
        
        self.locShuffle_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.locShuffle_gb1)
        
        widget.setContents(layout)
        
        return widget
        
#    def placeFieldPage(self):
#        widget= ScrollableWidget()
#
#        self.placeField_gb1= addGroupBox_2("", "placeField_gb1")
#        
#        boxLayout= QtWidgets.QVBoxLayout()
#        boxLayout.addWidget(QtWidgets.QLabel("No parameter to set"))
#        self.placeField_gb1.setLayout(boxLayout)
#        
#        layout= QtWidgets.QVBoxLayout()
#        layout.addWidget(self.placeField_gb1)
#        
#        widget.setContents(layout)
#        return widget

    def locTimeLapsePage(self):
        widget= ScrollableWidget()

        self.locTimeLapse_gb1= addGroupBox_2("", "locTimeLapse_gb1")
        
        boxLayout= QtWidgets.QVBoxLayout()
        boxLayout.addWidget(QtWidgets.QLabel("No parameter to set"))
        self.locTimeLapse_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.locTimeLapse_gb1)
        
        widget.setContents(layout)
        return widget
        
    def locTimeShiftPage(self):
        widget= ScrollableWidget()
        
        self.locTimeShift_gb1= addGroupBox_2("Shift Specifications", "locTimeShift_gb1")
        
        self.locShiftMax= addSpinBox_2(1, 100, "locShiftMax")
        self.locShiftMax.setValue(10)
        
        self.locShiftMin= addSpinBox_2(-100, -1, "locShiftMin")
        self.locShiftMin.setValue(-10)
        
        self.locShiftStep= addSpinBox_2(1, 3, "locShiftStep")
        self.locShiftStep.setValue(1)

        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Maximum Shift", self.locShiftMax, "indices [range: 1 to 100]")
        boxLayout.addRow("Minimum Shift", self.locShiftMin, "indices [range: -1 to -100]")
        boxLayout.addRow("Index Steps", self.locShiftStep, "[1, 2, 3]")
        
        self.locTimeShift_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.locTimeShift_gb1)
        
        widget.setContents(layout)
        return widget
        
    def spatialCorrPage(self):
        widget= ScrollableWidget()
        # Box- 1
        self.spatialCorr_gb1= addGroupBox_2("2D Correlation", "spatialCorr_gb1")

        self.spatialCorrMinObs= addSpinBox_2(0, 100, "spatialCorrMinObs")
        self.spatialCorrMinObs.setValue(20)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Minimum No. of Valid Pixels", self.spatialCorrMinObs, "[range: 1-100]")
        
        self.spatialCorr_gb1.setLayout(boxLayout)

        # Box- 2        
        self.spatialCorr_gb2= addGroupBox_2("Rotational Correlation", "spatialCorr_gb2")

        self.rotCorrBin= addComboBox_2("rotCorrBin")
        rotCorrBinItems= [str(d) for d in range(1, 360) if 360 % d== 0 and d>=3 and d<=45]
        self.rotCorrBin.addItems(rotCorrBinItems)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Rotational Correlation Binsize", self.rotCorrBin, "degree")
        
        self.spatialCorr_gb2.setLayout(boxLayout)
        
#        # Box- 3
        self.spatialCorr_gb3= addGroupBox_2("Smoothing Box Kernal", "spatialCorr_gb3")
        
        self.spatialCorrFilter= addComboBox_2("spatialCorrFilter")
        self.spatialCorrFilter.addItems(["Box", "Gaussian"])
        
        self.spatialCorrKernLen= addSpinBox_2(1, 11, "spatialCorrKernLen")
        self.spatialCorrKernLen.setValue(5)
        self.spatialCorrKernLen.setSingleStep(2)
        # Change step size to 0.5 if Gaussian is selected
                
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Smoothing Filter", self.spatialCorrFilter, "")
        boxLayout.addRow("Correlation Pixels Num/Sigma", self.spatialCorrKernLen, \
        "[range: 1-11]\n\r Box: odds")
        
        self.spatialCorr_gb3.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.spatialCorr_gb1)
        layout.addWidget(self.spatialCorr_gb2)
        layout.addWidget(self.spatialCorr_gb3)
        
        widget.setContents(layout)        
        
        return widget
        
    def gridPage(self):
        widget= ScrollableWidget()
        
        self.grid_gb1= addGroupBox_2("Analyses Paramters", "grid_gb1")
        
        self.gridAngTol= addSpinBox_2(1, 5, "gridAngTol")
        self.gridAngTol.setValue(2)
                
        self.gridAngBin= addComboBox_2("gridAngBin")
        gridAngBinItems= [str(d) for d in range(1, 360) if 360 % d== 0 and d>=3 and d<=45]
        self.gridAngBin.addItems(gridAngBinItems)

        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Angular Tolerance", self.gridAngTol, "degree [range: 1 to 5]")
        boxLayout.addRow("Angular Binsize", self.gridAngBin, "degree")
        
        self.grid_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.grid_gb1)
        
        widget.setContents(layout)
        return widget
        
    def borderPage(self):
        widget= ScrollableWidget()
        
        self.border_gb1= addGroupBox_2("Analyses Paramters", "border_gb1")
        
        self.borderFiringThresh= addDoubleSpinBox_2(0, 1, "borderFiringThresh")
        self.borderFiringThresh.setValue(0.1)
        self.borderFiringThresh.setSingleStep(0.05)
                
        self.borderAngBin= addComboBox_2("borderAngBin")
        borderAngBinItems= [str(d) for d in range(1, 360) if 360 % d== 0 and d>=3 and d<=45]
        self.borderAngBin.addItems(borderAngBinItems)
        
        self.borderStairSteps= addSpinBox_2(4, 10, "borderStairSteps")
        self.borderStairSteps.setValue(5)

        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Firing Threshold", self.borderFiringThresh, "degree [range: 0 to 1, step: 0.05]")
        boxLayout.addRow("Angular Binsize", self.borderAngBin, "degree")
        boxLayout.addRow("Stair Plot Steps", self.borderStairSteps, "4 to 10")
        
        self.border_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.border_gb1)
        
        widget.setContents(layout)
        return widget
        
    def gradientPage(self):
        widget= ScrollableWidget()
        
        self.gradient_gb1= addGroupBox_2("Gompertz Function Parameters", "gradient_gb1")
        
        self.gradAsympLim= addDoubleSpinBox_2(0.1, 1, "gradAsympLim")
        self.gradAsympLim.setValue(0.25)
        self.gradAsympLim.setSingleStep(0.05)
                
        self.gradDisplaceLim= addDoubleSpinBox_2(0.1, 1, "gradDisplaceLim")
        self.gradDisplaceLim.setValue(0.25)
        self.gradDisplaceLim.setSingleStep(0.05)
        
        self.gradGrowthRateLim= addDoubleSpinBox_2(0.1, 1, "gradGrowthRateLim")
        self.gradGrowthRateLim.setValue(0.5)
        self.gradGrowthRateLim.setSingleStep(0.05)

        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Asymptote a \xb1 ", self.gradAsympLim, "*a [range 0.1 to 1, step: 0.05]")
        boxLayout.addRow("Displacement b \xb1 ", self.gradDisplaceLim, "*b [range 0.1 to 1, step: 0.05]")
        boxLayout.addRow("Growth Rate c \xb1 ", self.gradGrowthRateLim, "*c [range 0.1 to 1, step: 0.05]")
        
        self.gradient_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.gradient_gb1)
        
        widget.setContents(layout)
        return widget
        
    def mraPage(self):
        widget= ScrollableWidget()
        self.mra_gb1= addGroupBox_2("Analyses Paramters", "mra_gb1")        

        self.mraInterval= addDoubleSpinBox_2(0.1, 2, "mraInterval")
        self.mraInterval.setValue(0.1)
        self.mraInterval.setSingleStep(0.1)
        
        self.mraEpisode= addSpinBox_2(60, 300, "mraEpisode")
        self.mraEpisode.setValue(120)
        self.mraEpisode.setSingleStep(30)
        
        self.mraNoRep= addSpinBox_2(100, 2000, "mraNoRep")
        self.mraNoRep.setValue(1000)
        self.mraNoRep.setSingleStep(100)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Subsampling Interval", self.mraInterval, "sec [range: 0.1-1, step: 0.1]")
        boxLayout.addRow("Chunk Length for Regression", self.mraEpisode, "sec [range: 60-300, step: 30]")
        boxLayout.addRow("No of Replication", self.mraNoRep, "[range: 100-2000, step: 100]") 
        
        self.mra_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.mra_gb1)
        
        widget.setContents(layout)
        
        return widget
        
    def interDependPage(self):
        widget= ScrollableWidget()

        self.interDepend_gb1= addGroupBox_2("", "interDepend_gb1")
        
        boxLayout= QtWidgets.QVBoxLayout()
        boxLayout.addWidget(QtWidgets.QLabel("Uses the parameters from other anlyses"))
        self.interDepend_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.interDepend_gb1)
        
        widget.setContents(layout)
        return widget

        
    def lfpSpectrumPage(self):
        widget= ScrollableWidget()
        # Box- 1
        self.lfpSpectrum_gb1= addGroupBox_2("Pre-filter (Butterworth) Properties", "lfpSpectrum_gb1")
        
        self.lfpPreFiltLowCut= addDoubleSpinBox_2(0.1, 4, "lfpPreFiltLowCut")
        self.lfpPreFiltLowCut.setValue(1.5)
        self.lfpPreFiltLowCut.setSingleStep(0.1)
        
        self.lfpPreFiltHighCut= addSpinBox_2(10, 500, "lfpPreFiltHighCut")
        self.lfpPreFiltHighCut.setValue(40)
        self.lfpPreFiltHighCut.setSingleStep(5)
        
        self.lfpPreFiltOrder= addSpinBox_2(1, 20, "lfpPreFiltOrder")
        self.lfpPreFiltOrder.setValue(5)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Lower Cutoff Frequency", self.lfpPreFiltLowCut, "Hz [range: 0.1-4, step: 0.1]")
        boxLayout.addRow("Higher Cutoff Frequency", self.lfpPreFiltHighCut, "Hz [range: 10-500, step: 5]")
        boxLayout.addRow("Filter Order", self.lfpPreFiltOrder, "[range: 1-20]")
        
        self.lfpSpectrum_gb1.setLayout(boxLayout)
        
        # Box- 2        
        self.lfpSpectrum_gb2= addGroupBox_2("Spectrum Analysis (PWELCH) Settings", "lfpSpectrum_gb2")
        
        self.lfpPwelchSegSize= addDoubleSpinBox_2(0.5, 100, "lfpPwelchSegSize")
        self.lfpPwelchSegSize.setValue(2)
        self.lfpPwelchSegSize.setSingleStep(0.5)
        
        self.lfpPwelchOverlap= addDoubleSpinBox_2(0.5, 50, "lfpPwelchOverlap")
        self.lfpPwelchOverlap.setValue(1)
        self.lfpPwelchOverlap.setSingleStep(0.5)
        
        self.lfpPwelchNfft= addComboBox_2("lfpPwelchNfft")
        NfftItems= [str(2**d) for d in range(7, 14)]
        self.lfpPwelchNfft.addItems(NfftItems)
        self.lfpPwelchNfft.setCurrentIndex(3)
                
        self.lfpPwelchFreqMax= addSpinBox_2(10, 500, "lfpPwelchFreqMax")
        self.lfpPwelchFreqMax.setValue(40)
        self.lfpPwelchFreqMax.setSingleStep(5)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Segment Size", self.lfpPwelchSegSize, "sec [range: 0.5-100, step: 0.5]")
        boxLayout.addRow("Overlap", self.lfpPwelchOverlap, "sec [range: 0.5-50, step: 0.5]")
        boxLayout.addRow("NFFT", self.lfpPwelchNfft, "[range: 128-8192, step: 128]")
        boxLayout.addRow("Maximum Frequency", self.lfpPwelchFreqMax, "Hz [range: 10-500, step: 5]")
        
        self.lfpSpectrum_gb2.setLayout(boxLayout)
        
        #        # Box- 3
        self.lfpSpectrum_gb3= addGroupBox_2("STFT Settings", "lfpSpectrum_gb3")
        
        self.lfpStftSegSize= addDoubleSpinBox_2(0.5, 100, "lfpStftSegSize")
        self.lfpStftSegSize.setValue(2)
        self.lfpStftSegSize.setSingleStep(0.5)
        
        self.lfpStftOverlap= addDoubleSpinBox_2(0.5, 50, "lfpStftOverlap")
        self.lfpStftOverlap.setValue(1)
        self.lfpStftOverlap.setSingleStep(0.5)
        
        self.lfpStftNfft= addComboBox_2("lfpStftNfft")
        NfftItems= [str(2**d) for d in range(7, 14)]
        self.lfpStftNfft.addItems(NfftItems)
        self.lfpStftNfft.setCurrentIndex(3)
        
        self.lfpStftFreqMax= addSpinBox_2(10, 500, "lfpStftFreqMax")
        self.lfpStftFreqMax.setValue(40)
        self.lfpStftFreqMax.setSingleStep(5)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Segment Size", self.lfpStftSegSize, "sec [range: 0.5-100, step: 0.5]")
        boxLayout.addRow("Overlap", self.lfpStftOverlap, "sec [range: 0.5-50, step: 0.5]")
        boxLayout.addRow("NFFT", self.lfpStftNfft, "[range: 128-8192, step: 128]")
        boxLayout.addRow("Maximum Frequency", self.lfpStftFreqMax, "Hz [range: 10-500, step: 5]")
        
        self.lfpSpectrum_gb3.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.lfpSpectrum_gb1)
        layout.addWidget(self.lfpSpectrum_gb2)
        layout.addWidget(self.lfpSpectrum_gb3)
        
        widget.setContents(layout)        
        
        return widget
    
    def spikePhasePage(self):
        
        widget= ScrollableWidget()
        
        self.spikePhase_gb1= addGroupBox_2("Analysis Parameters", "spikePhase_gb1")
                        
        self.phaseFreqMin= addDoubleSpinBox_2(1, 10, "phaseFreqMin")
        self.phaseFreqMin.setValue(6)
        self.phaseFreqMin.setSingleStep(0.5)
        
        self.phaseFreqMax= addDoubleSpinBox_2(8, 16, "phaseFreqMax")
        self.phaseFreqMax.setValue(12)
        self.phaseFreqMax.setSingleStep(0.5)        
        
        self.phasePowerThresh= addDoubleSpinBox_2(0, 1, "phasePowerThresh")
        self.phasePowerThresh.setValue(0.1)
        self.phasePowerThresh.setSingleStep(0.05)
        
        self.phaseAmpThresh= addDoubleSpinBox_2(0, 1, "phaseAmpThresh")
        self.phaseAmpThresh.setValue(0.15)
        self.phaseAmpThresh.setSingleStep(0.05)
        
        self.phaseBin= addComboBox_2("phaseBin")
        phaseBinItems= [str(d) for d in range(1, 360) if 360 % d== 0 and d>=5 and d<=45 ]
        self.phaseBin.addItems(phaseBinItems)
        
        self.phaseRasterBin= addSpinBox_2(1, 15, "phaseRasterBin")
        self.phaseRasterBin.setValue(2)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Frequency of Interest (Min)", self.phaseFreqMin, "Hz [range: 1-10, step: 0.5]")
        boxLayout.addRow("Frequency of Interest (Max)", self.phaseFreqMax, "Hz [range: 1-10, step: 0.5]")
        boxLayout.addRow("Band to Total Power Ratio (Min)", self.phasePowerThresh, "[range: 0-1, step: 0.05]")
        boxLayout.addRow("Segment to Overall Amplitude Ratio (Min)", self.phaseAmpThresh, "[range: 0-1, step: 0.05]")
        boxLayout.addRow("Phase Plot Binsize", self.phaseBin, "degree")
        boxLayout.addRow("Phase Raster Plot Binsize", self.phaseRasterBin, "degree[range: 1-15]")
        
        self.spikePhase_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.spikePhase_gb1)
        
        widget.setContents(layout)
        
        return widget
        
    def phaseLockPage(self):
        widget= ScrollableWidget()
        
        self.phaseLock_gb1= addGroupBox_2("Analysis Parameters", "phaseLock_gb1")
        
        self.phaseLockWinLow= addDoubleSpinBox_2(-1, -0.1, "phaseLockWinLow")
        self.phaseLockWinLow.setValue(-0.4)
        self.phaseLockWinLow.setSingleStep(0.05)
        
        self.phaseLockWinUp= addDoubleSpinBox_2(0.1, 1, "phaseLockWinUp")
        self.phaseLockWinUp.setValue(0.4)
        self.phaseLockWinUp.setSingleStep(0.05)

        self.phaseLockNfft= addComboBox_2("phaseLockNfft")
        NfftItems= [str(2**d) for d in range(7, 14)]
        self.phaseLockNfft.addItems(NfftItems)
        self.phaseLockNfft.setCurrentIndex(3)
        
        self.phaseLockFreqMax= addSpinBox_2(10, 500, "phaseLockFreqMax")
        self.phaseLockFreqMax.setValue(40)
        self.phaseLockFreqMax.setSingleStep(5)
        
        boxLayout= ParamBoxLayout()
        boxLayout.addRow("Analysis Window (Lower)", self.phaseLockWinLow, "sec [range: -1 to -0.1, step: 0.05]")
        boxLayout.addRow("Analysis Window (Upper)", self.phaseLockWinUp, "sec [range: 0.1 to 1, step: 0.05]")
        boxLayout.addRow("NFFT", self.phaseLockNfft, "[range: 128-8192, step: 128]")
        boxLayout.addRow("Frequency of Interest (Max)", self.phaseLockFreqMax, "Hz [range: 1-10, step: 0.5]")
        
        self.phaseLock_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.phaseLock_gb1)
        
        widget.setContents(layout)
        
        return widget
        print('Wait')
    def lfpSpikeCausalityPage(self):
        widget= ScrollableWidget()

        self.causality_gb1= addGroupBox_2("", "causality_gb1")
        
        boxLayout= QtWidgets.QVBoxLayout()
        boxLayout.addWidget(QtWidgets.QLabel("Analysis not implemented yet!"))
        self.causality_gb1.setLayout(boxLayout)
        
        layout= QtWidgets.QVBoxLayout()
        layout.addWidget(self.causality_gb1)
        
        widget.setContents(layout)
        return widget
        
class ScrollableWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.parentLayout = QtWidgets.QVBoxLayout(self)
        self.contWidget= QtWidgets.QWidget()
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setWidgetResizable(False)
    def setContents(self, contLayout):
        self.contLayout= contLayout
        self.contWidget.setLayout(self.contLayout)
        self.scrollArea.setWidget(self.contWidget)
        self.parentLayout.addWidget(self.scrollArea)
        self.setLayout(self.parentLayout)
        
class ParamBoxLayout(QtWidgets.QVBoxLayout):
    def __int__(self):
        super().__init__()
    def addRow(self, label_1, widg, label_2):
        widg.resize(widg.sizeHint())
        hLayout= QtWidgets.QHBoxLayout()
        hLayout.addWidget(QtWidgets.QLabel(label_1), 0, QtCore.Qt.AlignLeft)
        hLayout.addWidget(widg, 0, QtCore.Qt.AlignLeft)
        hLayout.addWidget(QtWidgets.QLabel(label_2), 0, QtCore.Qt.AlignLeft)
        self.addLayout(hLayout)        



class Ui_getfiles(QtWidgets.QDialog):
    DOWN= 1
    UP= -1
    def __init__(self, parent= None, filters= ['.pdf', '.ps']):
        super().__init__(parent)
        self.parent= parent
        self.filters= filters
        self.current_filter= None
        self.files= []
        self.dir_icon= self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon)
        self.file_icon= self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)        
    def setupUi(self):
        self.setObjectName(_fromUtf8("getFilesWindow"))
        self.setEnabled(True)
        self.setFixedSize(680, 400)
        self.setWindowTitle(QtWidgets.QApplication.translate("getFilesWindow", "Select files", None))            
        
        self.dirList= QtWidgets.QListView(self)
        self.dirList.setObjectName(_fromUtf8("dirList"))
        self.dirList.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        
        self.dirModel= QtGui.QStandardItemModel(self.dirList)        
        self.dirList.setModel(self.dirModel)
        
        self.fileList= QtWidgets.QListView(self)
        self.fileList.setObjectName(_fromUtf8("fileList"))        
        self.fileList.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.fileModel= QtGui.QStandardItemModel(self.fileList)        
        self.fileList.setModel(self.fileModel)
        
        button_layout= QtWidgets.QVBoxLayout()
        self.addButton= addPushButton_2("Add", "addButton")
        self.removeButton= addPushButton_2("Remove", "removeButton")
        self.upButton= addPushButton_2("Move Up", "upButton")
        self.downButton= addPushButton_2("Move Down", "downButton")
        self.doneButton= addPushButton_2("Done", "doneButton")
        self.cancelButton= addPushButton_2("Cancel", "cancelButton")
        
        button_layout.addWidget(self.addButton)
        button_layout.addWidget(self.removeButton)
        button_layout.addWidget(self.upButton)
        button_layout.addWidget(self.downButton)
        button_layout.addWidget(self.doneButton)
        button_layout.addWidget(self.cancelButton)

        box_layout= QtWidgets.QVBoxLayout()
        
        self.filterLabel= addLabel_2("File Type", "modeLabel")
        self.filterBox= addComboBox_2("filterBox")
        self.filterBox.addItems(self.filters)
        
        self.folderLabel= addLabel_2("Current Folder", "folderLabel")
        self.folderLine = addLineEdit_2(os.getcwd(), "folderLine")
        
        self.dirBox= addComboBox_2("dirBox")
        self.dirBox.addItems(os.getcwd().split(os.sep))
        self.dirBox.setCurrentIndex(self.dirBox.findText(os.getcwd().split(os.sep)[-1]))
        self.backButton= addPushButton_2("Back", "backButton")
        
        dir_layout= QtWidgets.QHBoxLayout()
        dir_layout.addWidget(self.dirBox)
        dir_layout.addWidget(self.backButton)
        dir_layout.addStretch()

        box_layout.addWidget(self.filterLabel)
        box_layout.addWidget(self.filterBox)
        box_layout.addWidget(self.folderLabel)
        box_layout.addWidget(self.folderLine)
        box_layout.addLayout(dir_layout)
        
        bottom_layout= QtWidgets.QHBoxLayout()        
        bottom_layout.addWidget(self.dirList)
        bottom_layout.addLayout(button_layout)
        bottom_layout.addWidget(self.fileList)
        
        main_layout= QtWidgets.QVBoxLayout()
        main_layout.addLayout(box_layout)
        main_layout.addLayout(bottom_layout)
        
        self.setLayout(main_layout)
    
    def behaviourUi(self):
       self.filterBox.currentIndexChanged[str].connect(self.filter_changed)
       self.folderLine.textEdited[str].connect(self.line_edited)
       self.dirBox.currentIndexChanged[int].connect(self.dir_changed) 
       self.backButton.clicked.connect(self.hierarchy_changed)
       self.dirList.activated[QtCore.QModelIndex].connect(self.item_activated)
       self.addButton.clicked.connect(self.add_items)
       self.removeButton.clicked.connect(self.remove_items)
       self.upButton.clicked.connect(partial(self.move_items, 'up'))
       self.downButton.clicked.connect(partial(self.move_items, 'down'))
       self.doneButton.clicked.connect(self.done) 
       self.cancelButton.clicked.connect(self.close)
                
       self.folderLine.setText(os.getcwd())
       self.line_edited(os.getcwd())
       
    def filter_changed(self, value):
        self.current_filter= value
        self.update_list(self.folderLine.text())
    def line_edited(self, value):
       if os.path.exists(value):
           self.dirBox.clear()
           self.dirBox.addItems(value.split(os.sep))
           self.dirBox.setCurrentIndex(self.dirBox.findText(value.split(os.sep)[-1]))
    
    def dir_changed(self, value):
        directory= os.sep.join([self.dirBox.itemText(i) for i in range(value+ 1)])
        if os.sep not in directory:
            directory+= os.sep
        self.folderLine.setText(directory)
        self.update_list(directory)
        
    def update_list(self, directory):
        self.dirModel.clear()
        if os.path.isdir(directory):
            dir_content= os.listdir(directory)
            for f in dir_content:
                if os.path.isdir(os.path.join(directory, f)):
                    item= QtGui.QStandardItem(self.dir_icon, f)
                    item.setEditable(False)
                    self.dirModel.appendRow(item)
                    
            for f in dir_content:
                if os.path.isfile(os.path.join(directory, f)) and f.endswith(self.filterBox.currentText()):
                    item= QtGui.QStandardItem(self.file_icon, f)     
                    item.setEditable(False)
                    self.dirModel.appendRow(item)
                   
    def hierarchy_changed(self):
        curr_ind= self.dirBox.currentIndex()-1
        if curr_ind>= 0:
            self.dirBox.setCurrentIndex(curr_ind)
    def item_activated(self, qind):
        data= self.dirModel.itemFromIndex(qind).text()
        directory= os.path.join(self.folderLine.text(), data)
        if os.path.isdir(directory):
            self.folderLine.setText(directory) # setText() does not invoke textEdited(), manually calling line_eidted()
            self.line_edited(directory)
        
    def add_items(self):        
        qind= self.dirList.selectedIndexes()
        items= [self.dirModel.itemFromIndex(i) for i in qind]
        curr_files= [self.fileModel.item(i).text() for i in range(self.fileModel.rowCount())]
        
        for it in items:
            if os.path.isfile(os.path.join(self.folderLine.text(), it.text())) \
                                        and it.text().endswith(self.filterBox.currentText()):
                if not curr_files or it.text() not in curr_files: 
                    self.fileModel.appendRow(it.clone())
                                            
        self.fileList.setModel(self.fileModel)
    
    def remove_items(self): 
        qind= self.fileList.selectedIndexes()
        rows= [i.row() for i in qind][::-1]
        for r in rows:
            self.fileModel.removeRow(r) # Rows change after each removal, so the one in the highest index are not deleted
        self.fileList.setModel(self.fileModel)
        
    def move_items(self, direction= 'down'): # -1= move up, +1= move down
        qind= self.fileList.selectedIndexes()
        rows= [i.row() for i in qind]        
        if direction== 'up':
            rows.sort(reverse= False)
            newRows= [r- 1 for r in rows]
        elif direction== 'down':
            rows.sort(reverse= True)
            newRows= [r+ 1 for r in rows]
        
        for i, row in enumerate(rows):
            if not (0<= newRows[i]<self.fileModel.rowCount()):
                continue
            rowItem= self.fileModel.takeRow(row)
            self.fileModel.insertRow(newRows[i], rowItem)

        self.fileList.setModel(self.fileModel)
    def done(self):
        self.files= [os.path.join(self.folderLine.text(), self.fileModel.item(i).text()) \
                for i in range(self.fileModel.rowCount())]
        self.close()
    def close(self):
        self.fileModel.clear()
        self.hide()
    def get_files(self):
        return self.files