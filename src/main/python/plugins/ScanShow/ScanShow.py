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

from plugins.ScanShow.Ui_ScanShow import Ui_ScanShow

class ScanShow(QtWidgets.QWidget, QtWidgets.QGraphicsScene, Ui_ScanShow):
    """
        ScanShow display the events list read
    """
    def __init__(self, info_subjects, *args, **kwargs):
        super(ScanShow, self).__init__(*args, **kwargs)

        # Set up the user interface from Designer.
        self.setupUi(self)

        self.info_subjects = info_subjects.subjects
        self.subjects = list(self.info_subjects.keys())
        self.index_subject = 0
        self.subject = self.subjects[self.index_subject]

        for key in self.info_subjects.keys():
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
        self.result_layout.scale(1.3, 1.3)
        self._update_subject_info()
        self._update_scan_info()

    def _on_changed_view(self):      
        self._plot_det_info()

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


        self._plot_det_info()

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

        self._plot_det_info()

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


    def _plot_det_info( self):
        """ 
        Plot eeg signal and detection info.

        Parameters
        -----------
            signals    : Dictionnary of SignalModel
                A dictionary of channels with SignalModel with properties :
                name:          The name of the channel
                samples:        The samples of the signal
                alias:          The alias of the channel
                sample_rate:    The sample rate of the signal
                start_time:     The start time of the recording

        """
        if hasattr(self, 'scene'):
            # Manage the figure
            self.scene.clear() # reset the hold on 

            if self.image_radioButton.isChecked():
                pix = QPixmap(self.info_subjects[self.subject].scans[self.scan].image.filename)
                item = QtWidgets.QGraphicsPixmapItem(pix)
                self.scene.addItem(item)
                self.result_layout.setScene(self.scene)
                self.result_layout.centerOn(0,0)
            elif self.pixel_radioButton.isChecked():
                pass

            self.scene.update()

        # #----------------------------------------------------------------------
        # # Plot eeg signal
        # n_chan = len(signals)
        # gs = self.figure.add_gridspec(n_chan, hspace=0)
        # ax1 = gs.subplots(sharex=True, sharey=False)
        # chan_sel = 0

        # for signal in signals:
        #     fs = signals[signal].sample_rate
        #     chan_name = signals[signal].name
        #     time_vect = np.linspace(0, self.duration, num = int(fs*self.duration))

        #     if n_chan>1:
        #         ax1[chan_sel].plot(time_vect, signals[signal].samples, 'b', linewidth=1, alpha=0.75)
        #         if not len(self.events_index) == 0:
        #             for index in self.events_index:
        #                 start = (self.events['start_sec'][index] - self.start)
        #                 end = start + (self.events['duration_sec'][index])
        #                 time_vect_event = np.linspace(start, end, num = int(fs*self.events['duration_sec'][index]))
        #                 ax1[chan_sel].plot(time_vect_event, signals[signal].samples[int(start*fs):int(end*fs)], 'r', linewidth=1, alpha=0.75)

        #     else:
        #         ax1.plot(time_vect, signals[signal].samples, 'b', linewidth=1, alpha=0.75)
        #         if not len(self.events_index) == 0:
        #             for index in self.events_index:
        #                 start = (self.events['start_sec'][index] - self.start)
        #                 end = start + (self.events['duration_sec'][index])
        #                 time_vect_event = np.linspace(start, end, num = int(fs*self.events['duration_sec'][index]))
        #                 ax1.plot(time_vect_event, signals[signal].samples[int(start*fs):int(end*fs)], 'r', linewidth=1, alpha=0.75)

        #     # Add vertical lines for sec
        #     nsec = int(self.duration)
        #     for sec_i in range(nsec):
        #         if n_chan>1:
        #             ax1[chan_sel].vlines(x=sec_i, ymin=min(signals[signal].samples),\
        #                 ymax=max(signals[signal].samples), linewidth=0.5, color='b', linestyles='--') 
        #         else:
        #             ax1.vlines(x=sec_i, ymin=min(signals[signal].samples),\
        #                  ymax=max(signals[signal].samples), linewidth=0.5, color='b', linestyles='--')                     

        #     if n_chan>1:
        #         ax1[chan_sel].set_ylabel(chan_name, loc='center', rotation=0, labelpad=30)
        #         ax1[chan_sel].set_xlabel('time [s]')
        #         ax1[chan_sel].set_xlim((time_vect[0], time_vect[-1]))
        #         # Turn off tick labels
        #         ax1[chan_sel].set_yticklabels([])
        #     else:
        #         ax1.set_ylabel(chan_name, loc='center', rotation=0, labelpad=30)
        #         ax1.set_xlabel('time [s]')
        #         ax1.set_xlim((time_vect[0], time_vect[-1]))
        #         # Turn off tick labels
        #         ax1.set_yticklabels([])
        #     chan_sel += 1

        # # Hide x labels and tick labels for all but bottom plot.
        # if n_chan>1:
        #     for ax in ax1:
        #         ax.label_outer()

        # # Add suptitle
        # self.figure.suptitle(signals[signal].alias + ' From Events')
        # # Redraw the figure, needed when the show button is pressed more than once
        # self.canvas.draw()


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

        # # Plot eeg signal.
        #     if self.signals_radioButton.isChecked():
        #         self._plot_det_info(self.signal)
        #     elif self.delta_signals_radioButton.isChecked():
        #         delta_signal = {}
        #         for channel in self.prev_signal:
        #             delta_signal[channel] = self.prev_signal[channel].clone(clone_samples=True)
        #             delta_signal[channel].samples = delta_signal[channel].samples - self.signal[channel].samples
        #         self._plot_det_info(delta_signal)
