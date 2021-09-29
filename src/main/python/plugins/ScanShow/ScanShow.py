#! /usr/bin/env python3

"""
    Results viewer of the ScanShow plugin
"""

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap
from PIL import Image, ImageQt

from plugins.ScanShow.Ui_ScanShow import Ui_ScanShow

class ScanShow(QtWidgets.QWidget, QtWidgets.QGraphicsScene, Ui_ScanShow):
    """
        ScanShow display the events list read
    """
    def __init__(self, info_subjects, subjects, images_as_array, *args, **kwargs):
        super(ScanShow, self).__init__(*args, **kwargs)

        # Set up the user interface from Designer.
        self.setupUi(self)

        self.info_subjects = info_subjects.subjects
        self.subjects = subjects
        self.index_subject = 0
        self.subject = self.subjects[self.index_subject]
        self.images_as_array = images_as_array

        for key in self.subjects:
            self.subject_comboBox.addItem(key)

        self.scans = list(self.info_subjects[self.subject].scans.keys())
        self.index_scan = 0
        self.scan = self.scans[self.index_scan]

        for key in self.info_subjects[self.subject].scans.keys():
            self.scan_comboBox.addItem(key)

        self.back_subject_pushButton.setEnabled(False)
        self.back_scan_pushButton.setEnabled(False)

        self.scene = QtWidgets.QGraphicsScene()
        self.scene.setSceneRect(0, 0, 0, 0)
        self.result_layout.setScene(self.scene)
        self.result_layout.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.result_layout.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.result_layout.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.result_layout.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.result_layout.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        self.result_layout.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.result_layout.centerOn(0,0)
        self._update_subject_info()
        self._update_scan_info()

    def _on_changed_view(self):      
        self._show_image()

    def _update_subject_info( self):

         # Fill info for subjects
        self.sid_lineEdit.setText(self.info_subjects[self.subject].sid)
        self.gender_lineEdit.setText(self.info_subjects[self.subject].gender)
        self.hand_lineEdit.setText(self.info_subjects[self.subject].hand)
        self.age_lineEdit.setText(str(self.info_subjects[self.subject].age))
        self.educ_lineEdit.setText(str(self.info_subjects[self.subject].educ))
        self.ses_lineEdit.setText(str(self.info_subjects[self.subject].ses))
        self.mmse_lineEdit.setText(str(self.info_subjects[self.subject].mmse))
        self.cdr_lineEdit.setText(str(self.info_subjects[self.subject].cdr))
        self.e_tiv_lineEdit.setText(str(self.info_subjects[self.subject].e_tiv))
        self.n_wbv_lineEdit.setText(str(self.info_subjects[self.subject].n_wbv))
        self.asf_lineEdit.setText(str(self.info_subjects[self.subject].asf))
        self.delay_lineEdit.setText(str(self.info_subjects[self.subject].delay))

        self.subject_comboBox.setCurrentText(self.subject)
        self._show_image()

    def _update_scan_info( self):

         # Fill info for subjects
        self.scan_number_lineEdit.setText(self.info_subjects[self.subject].scans[self.scan].scan_number)
        self.scan_type_lineEdit.setText(self.info_subjects[self.subject].scans[self.scan].scan_type)
        self.vox_res_lineEdit.setText(self.info_subjects[self.subject].scans[self.scan].vox_res)
        self.rect_fov_lineEdit.setText(self.info_subjects[self.subject].scans[self.scan].rect_fov)
        self.orientation_lineEdit.setText(self.info_subjects[self.subject].scans[self.scan].orientation)
        self.tr_lineEdit.setText(str(self.info_subjects[self.subject].scans[self.scan].tr))
        self.te_lineEdit.setText(str(self.info_subjects[self.subject].scans[self.scan].te))
        self.ti_lineEdit.setText(str(self.info_subjects[self.subject].scans[self.scan].ti))
        self.flip_lineEdit.setText(str(self.info_subjects[self.subject].scans[self.scan].flip))

        self.scan_comboBox.setCurrentText(self.scan)
        self._show_image()

    def on_subject_changed( self):
        self.subject = self.subject_comboBox.currentText()
        self.index_subject = self.subject_comboBox.currentIndex()

        self._on_navigate_subject()
        if hasattr(self, 'index_scan'):
            self._on_navigate_scan()

    def on_scan_changed( self):
        self.scan = self.scan_comboBox.currentText()
        self.index_scan = self.scan_comboBox.currentIndex()
        
        self._on_navigate_scan()

    def _show_image( self):
        if hasattr(self, 'scene'):
            # Manage the figure
            self.scene.clear() # reset the hold on 

            index_array = 0
            for sub in self.subjects:
                for sc in self.info_subjects[sub].scans:
                    if sub == self.subject and sc == self.scan:
                        break
                    index_array +=1
                if sub == self.subject and sc == self.scan:
                        break

            img = Image.fromarray(np.uint8(np.reshape(self.images_as_array[index_array],(256,256)) * 255) , 'L')
            w, h = img.size
            view_size = self.result_layout.size()
            ratio = min(view_size.width()/w, view_size.height()/h)
            new_img = img.resize((int(w * ratio), int(h * ratio)), Image.NEAREST)
            self.imgQ = ImageQt.ImageQt(new_img)  # we need to hold reference to imgQ, or it will crash
            pixMap = QPixmap.fromImage(self.imgQ)
            self.scene.addPixmap(pixMap)

            self.scene.update()

    def on_next_subject( self):
        """A slot called when >> button is pressed by the user.
        The user wants to display the following window.
        """  
        self._on_navigate_subject(next_on_sub=1)

    def on_prev_subject( self):
        """A slot called when << button is pressed by the user.
        The user wants to display the previous window.
        """    
        self._on_navigate_subject(next_on_sub=-1)

    def on_next_scan( self):
        """A slot called when >> button is pressed by the user.
        The user wants to display the following window.
        """  
        self._on_navigate_scan(next_on_scan=1)

    def on_prev_scan( self):
        """A slot called when << button is pressed by the user.
        The user wants to display the previous window.
        """    
        self._on_navigate_scan(next_on_scan=-1)

    def _on_navigate_subject(self, next_on_sub=0):
        ''' Call when the user presses the >>, << or enter after editing the subject.
        '''

        # Change index of event
        self.index_subject = self.index_subject + next_on_sub
        self.subject = self.subjects[self.index_subject]

        if hasattr(self, 'index_scan'):
            self.index_scan = 0
            self.scan = self.scans[self.index_scan]

        # Desable the button if its impossible to press the button another time.
        if self.index_subject - 1 < 0:
            self.back_subject_pushButton.setEnabled(False)
        else:
            self.back_subject_pushButton.setEnabled(True)
        if self.index_subject + 1 > (len(self.subjects) - 1):
            self.next_subject_pushButton.setEnabled(False)
        else:
            self.next_subject_pushButton.setEnabled(True)
        
        # Update event
        self._update_subject_info()

    def _on_navigate_scan(self, next_on_scan=0):
        ''' Call when the user presses the >>, << or enter after editing the subject.
        '''

        # Change index of event
        self.index_scan = self.index_scan + next_on_scan
        self.scan = self.scans[self.index_scan]

        # Desable the button if its impossible to press the button another time.
        if self.index_scan - 1 < 0:
            self.back_scan_pushButton.setEnabled(False)
        else:
            self.back_scan_pushButton.setEnabled(True)
        if self.index_scan + 1 > (len(self.scans) - 1):
            self.next_scan_pushButton.setEnabled(False)
        else:
            self.next_scan_pushButton.setEnabled(True)
        
        # Update event
        self._update_scan_info()
