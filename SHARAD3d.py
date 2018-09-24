# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SHARAD3d
                                 A QGIS plugin
 SHARAD3d
                              -------------------
        begin                : 2018-02-01
        git sha              : $Format:%H$
        copyright            : (C) 2018 by SITING XIONG
        email                : siting.xiong.14@ucl.ac.uk
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, QFileInfo
#from PyQt4.QtGui import QAction, QIcon  # comment by XST on 01-Feb-2018
# Initialize Qt resources from file resources.py
import resources
# Import the code for the dialog
from SHARAD3d_dialog import SHARAD3dDialog
### Comment by XST
#import os.path

### add by XST on 01-Feb-2018
from PyQt4.QtCore import Qt
from qgis.core import *
from PyQt4.QtGui import *
import processing
import urllib
import subprocess
from SHARAD import SHARAD2d
from SHARAD.sharadPtcloud import *
import os
# import qt_embedding
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import gdal, gdalnumeric
#import myavi_embedding
### add by XST on 01-Feb-2018

class SHARAD3d:
    
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'SHARAD3d_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)


        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&SHARAD3d')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'SHARAD3d')
        self.toolbar.setObjectName(u'SHARAD3d')

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('SHARAD3d', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        # Create the dialog (after translation) and keep reference
        self.dlg = SHARAD3dDialog()

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/SHARAD3d/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'SHARAD3d'),
            callback=self.run,
            parent=self.iface.mainWindow())
        
        ### add by XST on 1-Feb-2018
        self.dockOpened = False		#remember for not reopening dock if there's already one opened
        #for le in self.dlg.findChildren(QLineEdit):
        #   le.setText("")
        
        self.fulldatalayer = None
        self.outputclip = None
        self.datafolder = None
        self.resultfolder = None
        self.dtmfile = None
        self.datalist = None
    
        self.dir = os.path.dirname(__file__)

        self.frame = QWidget()
        #self.frame.setGeometry(0, 0, 500, 400)
        #self.frame.setWindowTitle('SHARAD Results')
        #self.frame.hbox = QHBoxLayout()
        #self.frame.hbox.addStretch(1)
        
        # Initial the self.form
        #self.form = viewApp.AppForm()
        #self.frame = gl.GLViewWidget()
        
        # Don't create a new QApplication, it would unhook the Events
        # set by Traits on the existing QApplication. Simply use the
        # '.instance()' method to retrieve the existing one.
        #self.frame = QWidget()
        #self.frame.setWindowTitle("Embedding Mayavi in a PyQt4 Application")
        
        ### add by XST on 1-Feb-2018
    
    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&SHARAD3d'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar


    def run(self):
        """Run method that performs all the real work"""
        
        ### add by XST on 1-Feb-2018
        if self.dockOpened == False:
            
            # show the dialog
            self.dlg.show()

            layers = self.iface.legendInterface().layers()
            veclayer_list = []
            raslayer_list = []
            for layer in layers:
                if layer.type() == layer.VectorLayer:
                    veclayer_list.append(layer.name())
                else:
                    raslayer_list.append(layer.name())
        
            # actions for comboBoxes 1,2,3 which are used for loading layers
            self.dlg.comboBox.clear()
            self.dlg.comboBox_2.clear()
            self.dlg.comboBox_3.clear()
            self.dlg.comboBox.addItems(veclayer_list)
            self.dlg.comboBox_2.addItems(veclayer_list)
            self.dlg.comboBox_3.addItems(raslayer_list)

            # actions for pushButtons to open, save files and point folder
            self.dlg.pushButton.clicked.connect(lambda: self.saveFile(self.dlg.lineEdit)) # trigger save file dialog
            self.dlg.pushButton_2.clicked.connect(lambda: self.selectFolder(self.dlg.lineEdit_2)) # trigger select folder dialog
            
            # actions for showing the file or folder path
            self.dlg.lineEdit_2.textChanged.connect(self.setDatafolder)

            # actions for pushButtons in ROI & Downloading panel
            self.dlg.pushButton_3.clicked.connect(self.clipLayer)
            self.dlg.pushButton_4.clicked.connect(self.download)
            #self.dlg.pushButton_5.clicked.connect(self.projectGeom)
            
            # actions for pushButtons to process 2-D radargrams
            self.dlg.pushButton_13.clicked.connect(self.process2d)
            self.dlg.pushButton_14.clicked.connect(lambda: self.process2d('all'))
            
            # actions for pushButtons in listWidget for selecting files
            self.dlg.pushButton_6.clicked.connect(self.removefromList)
            self.dlg.pushButton_7.clicked.connect(self.addback2List)
            self.dlg.pushButton_10.clicked.connect(lambda: self.removefromList('all'))
            self.dlg.pushButton_11.clicked.connect(lambda: self.addback2List('all'))
            
            
            self.dlg.pushButton_8.clicked.connect(self.ptCloud)
            self.dlg.pushButton_9.clicked.connect(self.subLayers)
            self.dlg.pushButton_12.clicked.connect(self.interpolatePoints)
            self.dlg.pushButton_15.clicked.connect(self.loadList)
            self.dlg.pushButton_16.clicked.connect(self.saveList)
            
            self.dlg.listWidget.itemDoubleClicked.connect(self.loadImage)

            self.dlg.horizontalSlider.valueChanged.connect(self.sliderText)
            self.dlg.horizontalSlider_2.valueChanged.connect(self.sliderText)
            self.dlg.horizontalSlider_3.valueChanged.connect(self.sliderText)
            self.dlg.horizontalSlider_4.valueChanged.connect(self.sliderText)

            self.dockOpened = True
            ### add by XST on 1-Feb-2018
        
        # Run the dialog event loop
        result = self.dlg.exec_()
        
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass

    ### add by XST on 1-Feb-2018
#************************************* Browse pushButton actions ******************************************
    def openFile(self, lineEdit):
        prjpath = QgsProject.instance().fileName()
        if prjpath == '':
            prjpath = '/Users'
        lineEdit.setText(QFileDialog.getOpenFileName(self.dlg, 'Open file',prjpath,"Files (*.shp *.txt *.tiff *.tif)"))
            
    def saveFile(self, lineEdit):
        prjpath = QgsProject.instance().fileName()
        if prjpath == '':
            prjpath = '/Users'
        lineEdit.setText(QFileDialog.getSaveFileName(self.dlg, 'Save File', prjpath))

    def selectFolder(self, lineEdit):
        prjpath = QgsProject.instance().fileName()
        if prjpath == '':
            prjpath = '/Users'
        lineEdit.setText(QFileDialog.getExistingDirectory(self.dlg,'Select directory'))

#************************************* Workspace changed action ******************************************

    def bash_command(self, cmd):
        p = subprocess.Popen(cmd, shell= True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #output, error = p.communicate()
    
    def setDatafolder(self):
        
        self.datafolder = self.dlg.lineEdit_2.text()
        if self.datafolder.endswith('/'):
            self.datafolder = self.datafolder[:-1]
        
        try:
            with open(self.datafolder + '/datalist.txt', 'r') as file:
                self.datalist = file.read().splitlines()

            # create folders
            tmpstr = self.datafolder + '/Results'
            if not os.path.isdir(tmpstr):
                os.makedirs(tmpstr)

            tmpstr = self.datafolder + '/Radargrams'
            if not os.path.isdir(tmpstr):
                os.makedirs(tmpstr)

        except IOError:
            self.datalist = None

    def clipLayer(self):
        # check comboBoxes has selected indices
        if self.dlg.comboBox.currentText() == '':
            return
        if self.dlg.comboBox_2.currentText() == '':
            return

        # add layers
        for layer in self.iface.legendInterface().layers():
            if layer.name() == self.dlg.comboBox.currentText():
                layer1 = layer
            if layer.name() == self.dlg.comboBox_2.currentText():
                layer2 = layer
            
        if layer1 is None or layer2 is None:
            w = QWidget()
            # warning, information, critical
            QMessageBox.information(w, "Error", 'None')
            w.show()

        self.dlg.textBrowser.append('Clipping %s with %s' %(layer2.name(),layer1.name()))
        processing.runalg("qgis:clip",layer2, layer1, self.dlg.lineEdit.text())


        # load the output shape file as a layer
        filestr = self.dlg.lineEdit.text()
        layername = filestr.split('/')[-1].split('.')[0]
        layer = self.iface.addVectorLayer(self.dlg.lineEdit.text(),layername,"ogr")
        if not layer:
            w = QWidget()
            # warning, information, critical
            QMessageBox.information(w, "Error", "Layer failed to load!")
            w.show()
    
        # get the output shape file and retrieve the attribute of product ID (index: 13)
        self.datalist = []
        idx = layer.fieldNameIndex('ProductId')
        for feature in layer.getFeatures():
            productid = str(feature.attributes()[idx])
            self.datalist.append(productid.split('_')[1])

        if self.datafolder is not None:

            with open(self.datafolder + '/datalist.txt', 'w') as file:
                file.writelines(["%s\n" % item for item in self.datalist])
        else:
            w = QWidget()
            # warning, information, critical
            QMessageBox.information(w, "Warning", "No assigned workspace.")
            w.show()
    
        self.loadList()


    def download(self):
        
        dataurl = 'http://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v1/mrosh_2001/data'
        browseurl = 'http://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v1/mrosh_2001/browse'
        
        if self.datalist is None:
            # clip the SHARAD transects by using the ROI of shape file
            self.clipLayer()
        cmd = "cp %s/bashscripts/batch_download_sharad.sh %s" %(self.dir, self.datafolder)
        self.bash_command(cmd)
        
        tmpfolder = self.datafolder + '/Radargrams'
        self.dlg.textBrowser.append('Please download data using bash script in the folder of workspace: bash batch_download_sharad.sh %s' %(tmpfolder))


#for id in self.datalist:
#            head = id[:4]
#            self.dlg.textBrowser.append('Downloading %s'%id)
            #urllib.urlretrieve(dataurl + '/rgram/s_' + head + 'xx/s_' + id + '_rgram.img', self.datafolder + '/radargrams/s_'+ id + '_rgram.img')
            #urllib.urlretrieve(dataurl + '/rgram/s_' + head + 'xx/s_' + id + '_rgram.lbl', self.datafolder +'/Radargrams/s_'+ id + '_rgram.lbl')
            #urllib.urlretrieve(dataurl + '/geom/s_' + head + 'xx/s_' + id + '_geom.tab', self.datafolder +'/Radargrams/s_'+ id + '_geom.tab')
            #urllib.urlretrieve(dataurl + '/geom/s_' + head + 'xx/s_' + id + '_geom.lbl', self.datafolder +'/Radargrams/s_'+ id + '_geom.lbl')
            #            urllib.urlretrieve(browseurl + '/tiff/s_' + head + 'xx/s_' + id + '_tiff.lbl', self.datafolder + '/Radargrams/s_' + id + '_tiff.lbl')
            #urllib.urlretrieve(browseurl + '/tiff/s_' + head + 'xx/s_' + id + '_tiff.tif', self.datafolder + '/Radargrams/s_' + id + '_tiff.tif')

    def projectGeom(self):
        
        # need to be sorted out, not working here
        tmpstr = self.datafolder + '/Radargrams'
        cmd = "if [ -d %s ]; then bash %s/bashscripts/batch_proj_geotab.sh %s; fi" %(tmpstr,self.dir,tmpstr)
        self.bash_command(cmd)

#******************************** actions for file list in listWidgets **************************************
    def loadList(self):
        '''
        # load the list file in workspace to listView
        
        #if self.datalist is None:
        #    self.datalist = []
        self.datafolder = self.dlg.lineEdit_2.text()
        if self.datafolder.endswith('/'):
            self.datafolder = self.datafolder[:-1]
        try:
            with open(self.datafolder + '/datalist.txt', 'r') as file:
                self.datalist = file.read().splitlines()
            
            # first clear the list widget
            self.dlg.listWidget.clear()
    
'''
        # first clear the list widget
        self.dlg.listWidget.clear()
        self.setDatafolder()
        if self.datalist is not None:
            try:
                for id in self.datalist:
                    item = QListWidgetItem(str(id))
                    self.dlg.listWidget.addItem(item)
                    self.dlg.listWidget.show()
            except IOError:
                w = QWidget()
                # warning, information, critical
                QMessageBox.information(w, "Error", "datalist.txt can't be loaded as list!")
                w.show()
        else:
            w = QWidget()
            # warning, information, critical
            QMessageBox.information(w, "Error", "No datalist.txt found in workspace!")
            w.show()

    def saveList(self):
        # save the re-arranged file list into a new file list
        itemsTextList =  [str(self.dlg.listWidget.item(i).text()) for i in range(self.dlg.listWidget.count())]
        if not itemsTextList:
            pass
        
        with open(self.datafolder + '/datalist-cleaned.txt', 'w') as file:
            file.writelines(["%s\n" % item for item in itemsTextList])

    def removefromList(self, allflag):
        # load the result into the image view (which needs to be added to tab2)
    
        if allflag == 'all':
            itemSelect =  [self.dlg.listWidget.item(i) for i in range(self.dlg.listWidget.count())]
        else:
            itemSelect = self.dlg.listWidget.selectedItems()
            if itemSelect is None: return

        for item in itemSelect:
            self.dlg.listWidget.takeItem(self.dlg.listWidget.row(item))
            self.dlg.listWidget_2.addItem(item)
            self.dlg.listWidget_2.sortItems()

    def addback2List(self, allflag):
        if allflag == 'all':
            itemSelect =  [self.dlg.listWidget_2.item(i) for i in range(self.dlg.listWidget_2.count())]
        else:
            itemSelect = self.dlg.listWidget_2.selectedItems()
            if itemSelect is None: return
    
        for item in itemSelect:
            self.dlg.listWidget_2.takeItem(self.dlg.listWidget_2.row(item))
            self.dlg.listWidget.addItem(item)
            self.dlg.listWidget.sortItems()

    def loadImage(self):

        # load image in bottom of the dialog
        itemSelect = self.dlg.listWidget.selectedItems()
        if itemSelect is None: return
        if len(itemSelect) > 1: return
        
        imageId = self.datafolder + '/Results/s_' + itemSelect[0].text() + '.png'
        
        #label = QLabel(self.frame)
        pixmap = QPixmap(imageId)
        #label.setPixmap(pixmap)
        self.dlg.label_3.setPixmap(pixmap.scaled(self.dlg.label_3.size(),Qt.KeepAspectRatio))
        #self.frame.show()

#*************************** actions for pushButtons to extract points **************************************
    def process2d(self,flagall):
        # input
        # self.datalist
        # self.datafolder
        # self.resultfolder
        # self.dtmfile
        # parameters: dtmwidth, gaborfitlers, minpts, mintrace
        # call the matlab function


        idArray = []
        if flagall == 'all':
            idArray = self.datalist
        else:
            for item in self.dlg.listWidget.selectedItems():
                idArray.append(item.text())

        roiName = self.dlg.comboBox.currentText()
        layers = self.iface.legendInterface().layers()
        roiSource = [ lyr.source() for lyr in layers if lyr.name() == roiName ]
        roiSource = roiSource[0]

        dtmName = self.dlg.comboBox_3.currentText()
        dtmSource = [ lyr.source() for lyr in layers if lyr.name() == dtmName ]
        dtmSource = dtmSource[0]

        self.dlg.textBrowser.append('Processing selected radargram within ROI of %s!' %roiSource)
        s = SHARAD2d.sharadProc()
        if os.path.isfile(roiSource):
            a = s.setROI(roiSource)
        if a == -1:
            self.dlg.textBrowser.append('Error polygon in %s!' %roiSource)
            return None
        
        
        if os.path.isfile(dtmSource):
            s.setDTM(dtmSource)

        mintrace = int(self.dlg.lineEdit_9.text())
        minpts = int(self.dlg.lineEdit_10.text())
        dtmwidth = int(self.dlg.lineEdit_4.text()) * 1000
        scale = [ int(i) for i in self.dlg.lineEdit_5.text().replace(';',',').split(',')]
        if len(scale) == 2:
            scale = xrange(scale[1]) + 1
        pfilter = [ int(i) for i in self.dlg.lineEdit_6.text().split(',')]
        params = [pfilter, scale, dtmwidth, minpts]
        print params
        
        for id in idArray:
            #process one radargram
            self.dlg.textBrowser.append('Processing %s' %id)
            s.setDataFolder(self.datafolder + '/Radargrams/')
            s.setID(id)
            s.readData()
            if s.geoinfo is None:
                continue
            # clipTrack according to study site
            # if there is no mintrace after clipping, function return None
            s.clipDatabyShape()
            if s.geoinfo is None or s.geoinfo.shape[0] < mintrace:
                continue
            s.chainProcess(params = params)
            self.dlg.textBrowser.append('Processing %s is successful!\nDouble Click product ID in list to view results...' %id)

    def ptCloud(self):
        # run bash script
        # cat all file into subpt txt file
        cmd = "cd %s; if [ -f subpt.txt ]; then rm subpt.txt; fi"%(self.datafolder + '/Results')
        self.bash_command(cmd)
        cmd = "cd %s; for i in $( ls *_subpt.txt ) do cat $i >> subpt.txt; done" %(self.datafolder + '/Results')
        self.bash_command(cmd)
    
    def sliderText(self):
        self.dlg.lineEdit_19.setText(str(self.dlg.horizontalSlider.value() * 0.01))
        self.dlg.lineEdit_20.setText(str(self.dlg.horizontalSlider_2.value()))
        self.dlg.lineEdit_21.setText(str(self.dlg.horizontalSlider_3.value()))
        self.dlg.lineEdit_22.setText(str(self.dlg.horizontalSlider_4.value()))

    def subLayers(self):
        # load data
        ptfile = self.datafolder + '/Results/subpt.txt'
        
        if not os.path.isfile(ptfile):
            return
        
        # get parameters
        dbeps = float(self.dlg.lineEdit_19.text())
        dbsamples = int(self.dlg.lineEdit_20.text())
        mindB = int(self.dlg.lineEdit_21.text())
        nlayers = int(self.dlg.lineEdit_22.text())

        outfolder = self.datafolder + '/Surface'
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)

        # The following three functions are imported from sharadPtcloud.py
        # clustering by DBSCAN, Hierarchical clustering

        clusterPoints(ptfile, outfolder, mindB, nlayers, dbeps, dbsamples)
        #data = []
        for i in xrange(1, nlayers + 1):
            fdata = np.loadtxt(outfolder + '/surface-%s.txt' % (i), delimiter = ',')
            #data.append(fdata)
            
            nclass = np.zeros((fdata.shape[0],1)) + i
            if i == 1:
                data = np.concatenate((fdata[:,:3], nclass), axis = 1)
            else:
                appenddata = np.concatenate((fdata[:,:3], nclass), axis = 1)
                data = np.concatenate((data, appenddata), axis = 0)
    
    
        #data[:,0] = data[:,0] - data[:,0].mean()
        #data[:,1] = data[:,1] - data[:,1].mean()
        #data[:,2] = data[:,2] - data[:,2].mean()
        data[:,2] = data[:,2] * 80
        
        w = gl.GLViewWidget()
        w.opts['distance'] = np.max((abs(data[:,0].mean()),abs(data[:,1].mean())))
        w.show()
        w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
        w.orbit(-90,-10)

        #C = pg.glColor('w')
        C = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,1],[1,1,0],[0,1,1],[1,1,1]])
        if nlayers > 7:
            C = np.random.rand(nlayers,3)
        #print C
        color = np.empty((data.shape[0],4))
        for i in xrange(nlayers):
            color[np.where(data[:,3] == i + 1)[0],:] = tuple(C[i,:]) + (1.0,)

        plt = gl.GLScatterPlotItem(pos = data[:,:3], size = 400, color = color, pxMode = False)
        plt.translate(-data[:,0].mean(), -data[:,1].mean(), -data[:,2].mean())
        w.addItem(plt)
        
        #self.form.setData(data)
        #self.form.on_draw()
        #self.form.show()
        
        #layout = QGridLayout(self.frame)
        #mayavi_widget = qt_embedding.MayaviQWidget(self.frame, data)
        #layout.addWidget(mayavi_widget, 1, 1)
        #self.frame.show()
        

    def interpolatePoints(self):
        
        if self.datafolder is None:
            return
        
        outfolder = self.datafolder + '/Surface'

        # Interpolate each layer by using TIN interpolation
        nlayers = int(self.dlg.lineEdit_22.text())
        xblock = int(self.dlg.lineEdit_13.text())
        yblock = int(self.dlg.lineEdit_14.text())
        kzthresh = int(self.dlg.lineEdit_16.text())
        kvthresh = int(self.dlg.lineEdit_15.text())
        res = int(self.dlg.lineEdit_17.text())
        smooth = int(self.dlg.lineEdit_18.text())

        for k in xrange(1,nlayers + 1):
            ptfile = outfolder + '/surface-%s.txt' %k
            outfile = outfolder + '/surface-%s_flt.txt' %k
            if not os.path.isfile(outfile):
                filterPoints(ptfile, outfile, xblock, yblock, kvthresh, kzthresh)
            ptfile = outfile
            outfile = outfolder + '/surface-%s_flt.tif' %k
            if not os.path.isfile(outfile):
                interpolatePoints(ptfile, outfile, res, smooth)
        1# find out which is the top surface layer
        mdtm = np.zeros(nlayers)
        for k in xrange(1,nlayers + 1):
            tiffile = outfolder + '/surface-%s_flt.tif' % k
            ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
            data = ds.GetRasterBand(1).ReadAsArray()
            maskeddata = np.ma.masked_invalid(data)
            mdtm[k-1] = maskeddata.mean()
            
        surfidx = np.argmax(mdtm) + 1
        tiffile = outfolder + '/surface-%s_flt.tif' %surfidx
        ds1 = gdal.Open(tiffile, gdal.GA_ReadOnly)
        data1 = ds1.GetRasterBand(1).ReadAsArray()
            
        for k in xrange(1, nlayers + 1):
            if k != surfidx:
                tiffile = outfolder + '/surface-%s_flt.tif' %k
                ds2 = gdal.Open(tiffile, gdal.GA_ReadOnly)
                data2 = ds2.GetRasterBand(1).ReadAsArray()
                
                driver = gdal.GetDriverByName('GTiff')
                outfile = outfolder + '/surface-%s_flt_depth.tif' %k
                dsOut = driver.Create(outfile, ds1.RasterXSize, ds1.RasterYSize, 1, gdal.GDT_Float32)
                gdalnumeric.CopyDatasetInfo(ds1,dsOut)
                dsOut.GetRasterBand(1).WriteArray((data1 - data2)/np.sqrt(3.14))
                dsOut.GetRasterBand(1).SetNoDataValue(-32768)
                dsOut = None
        ds1 = None

        for k in xrange(1,nlayers + 1):
            print k
            if k == surfidx:
                outfile = outfolder + '/surface-%s_flt.tif' %surfidx
            else:
                outfile = outfolder + '/surface-%s_flt_depth.tif' %k
                plotGeotiff(outfile)
            
            ### Load
            print outfile
            fileInfo = QFileInfo(outfile)
            path = fileInfo.filePath()
            baseName = fileInfo.baseName()

            layer = QgsRasterLayer(path, baseName)
            if layer.isValid() is True:
                QgsMapLayerRegistry.instance().addMapLayer(layer)

    ### add by XST on 1-Feb-2018
#************************************* Mouse listener actions ***********************************************
