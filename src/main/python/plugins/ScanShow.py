#! /usr/bin/env python3

"""
    Results viewer of the IcaRestore plugin
"""

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

from PyQt5 import QtWidgets
from PyQt5 import QtGui

from ui.Ui_MainWindow import Ui_MainWindow

class ScanShow(QtWidgets.QWidget, Ui_MainWindow):
    """
        IcaRestoreView display the events list read
    """
    def __init__(self, pixel_dict, *args, **kwargs):
        super(ScanShow, self).__init__(*args, **kwargs)
        self.figure = plt.imshow(np.array(pixel_dict.subjects['OAS1_0001_MR1'].scans['mpr-1'].pixels))
        # Add the figure tool bar
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)    
        # Add the figure into the result_layout
        self.result_layout.addWidget(toolbar)
        self.result_layout.addWidget(self.canvas)   
        

    def load_results(self):      
        
        # Set first window
        self.index_subject = 0
        self.index_scan = 0
        self.back_scan_pushButton.setEnabled(False)
        self.back_subject_pushButton.setEnabled(False)

        # Update event
        self._update_event_info()

        # # Plot first signal
        # if self.signals_radioButton.isChecked():
        #     self._plot_det_info(self.signal)
        # elif self.delta_signals_radioButton.isChecked():
        #     delta_signal = {}
        #     for channel in self.prev_signal:
        #         delta_signal[channel] = self.prev_signal[channel].clone(clone_samples=True)
        #         delta_signal[channel].samples = delta_signal[channel].samples - self.signal[channel].samples
        #     self._plot_det_info(delta_signal)

        # # Desable the button if its impossible to press the button another time.
        # if self.index - 1 < 0:
        #     self.prev_but.setEnabled(False)
        # else:
        #     self.prev_but.setEnabled(True)
        # if self.index + 1 > (len(self.signals) - 1):
        #     self.next_but.setEnabled(False)
        # else:
        #     self.next_but.setEnabled(True)

    def _update_event_info( self):
        pass
        #  # Fill info for first signal
        # self.event_lineEdit.setText(self.evnt)
        # self.duration_lineEdit.setText(str(self.duration))
        # self.event_index_lineEdit.setText(str(self.index))


    def on_event_index_changed( self):
        pass
        # if int(self.event_index_lineEdit.text()) >= 0 and int(self.event_index_lineEdit.text()) <= (len(self.signals) - 1):
        #     self.index = int(self.event_index_lineEdit.text())
        #     self._on_navigate()
        # else:
        #     self.event_index_lineEdit.setText(str(self.index))
        #     print("Error index outside of range")


    def _plot_det_info( self, signals):
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
        pass
        # # Manage the figure
        # self.figure.clear() # reset the hold on 

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


    def on_next_button( self):
        """A slot called when >> button is pressed by the user.
        The user wants to display the following window.
        """  
        self._on_navigate(next_on=1)


    def on_prev_button( self):
        """A slot called when << button is pressed by the user.
        The user wants to display the previous window.
        """    
        self._on_navigate(next_on=-1)


    def _on_navigate(self, next_on=0):
        ''' Call when the user presses the >>, << or enter after editing the time.
        '''

        # Change index of event
        self.index_subject = self.index_subject + next_on
        self.index_scan = self.index_scan + next_on
        
        self.duration = self.epochs_to_process.duration_sec[self.index]

        # Update event
        self._update_event_info()
        
        # # Desable the button if its impossible to press the button another time.
        # if self.index - 1 < 0:
        #     self.prev_but.setEnabled(False)
        # else:
        #     self.prev_but.setEnabled(True)
        # if self.index + 1 > (len(self.signals) - 1):
        #     self.next_but.setEnabled(False)
        # else:
        #     self.next_but.setEnabled(True)

        # # Plot eeg signal.
        #     if self.signals_radioButton.isChecked():
        #         self._plot_det_info(self.signal)
        #     elif self.delta_signals_radioButton.isChecked():
        #         delta_signal = {}
        #         for channel in self.prev_signal:
        #             delta_signal[channel] = self.prev_signal[channel].clone(clone_samples=True)
        #             delta_signal[channel].samples = delta_signal[channel].samples - self.signal[channel].samples
        #         self._plot_det_info(delta_signal)
