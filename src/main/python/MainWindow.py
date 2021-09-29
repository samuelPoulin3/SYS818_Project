#! /usr/bin/env python3

""" MainWindow of Classification algorithm

    This is the main window of the algorithm.
"""
import os
import sys

from PyQt5 import QtCore
from PyQt5 import QtWidgets
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QFileSystemModel, QFileDialog
from plugins.ImportSubjects import ImportSubjects
from ui.Ui_MainWindow import Ui_MainWindow
from plugins.ScanShow.ScanShow import ScanShow
from collections import defaultdict
from PyQt5.QtWidgets import QTableWidgetItem

DEBUG = False

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    stop_signal = QtCore.pyqtSignal()

    """Main Window."""
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # Setup the Ui_MainWindow generated by QtDesigner
        self.setupUi(self)
        self.model = QFileSystemModel()
        self.model.setRootPath(os.path.expanduser('~'))
        self.prepare_dataset.setVisible(False)
        ################### Remove this before final version #################
        self.subjects_path_lineEdit.setText("C:/Users/sampo/Python/PycharmProjects/SYS818_Project/Data/subjects")

        # Init internal variables
        self._subject_path = None
        self._use_multithread = True

    def on_new(self):
        """
        Blabla
        """
        pass

    def on_open(self):
        """
        Blabla
        """
        self._subject_path = QFileDialog.getExistingDirectory(
            self, 
            "Open Directory",
            "/home",
            QFileDialog.ShowDirsOnly
            | QFileDialog.DontResolveSymlinks)
        
        if self._subject_path == '' and self.subjects_path_lineEdit.text() == '':
            err_message = "ERROR MainWindow: no path selected"
            print(err_message)
        elif not self._subject_path == '' and self.subjects_path_lineEdit.text() == '':
            self.subjects_path_lineEdit.setText(self._subject_path)
        elif not self._subject_path == '' and not self.subjects_path_lineEdit.text() == '':
            self.subjects_path_lineEdit.setText(self._subject_path)

    def on_save(self):
        """
        Blabla
        """
        pass

    def on_exit(self):
        """ 
        Blabla
        """
        pass

    def on_run(self):
        """
        Blabla
        """
        self._subject_path = self.subjects_path_lineEdit.text()
        if self._use_multithread:
            self._start_multithread()
        self.info_subjects = ImportSubjects(self._subject_path)
        self.prepare_dataset.setVisible(True)
        self.subjects_found_lineEdit.setText(str(self.info_subjects.nb_images))            

    def reshape_data(self):
        self.subjects_train_pour = self.subjects_train_spinBox.value() / 100
        self.subjects_test_pour = self.subjects_test_spinBox.value() / 100
        if self.subjects_train_pour + self.subjects_test_pour > 1:
            err_message = "ERROR reshape_data pourcentage over 100%"
            print(err_message)

        self.number_epochs = self.number_epochs_spinBox.value()
        self.batch_size = self.batch_size_spinBox.value()

        self.label_table = {}
        self.label_table = {'Male': 0.0,
                            'Female': 1.0,
                            'Right': 0.0,
                            'Left': 1.0}

        train = 0
        self.labels = []
        self.use_train = []
        
        self.labels_train = []
        self.data_train = []
        subjects_train = int(self.info_subjects.nb_images * self.subjects_train_pour)
        for subject in self.info_subjects.subjects:
            self.use_train.append(subject)
            for scan in self.info_subjects.subjects[subject].scans:
                self.data_train.append([self.info_subjects.subjects[subject].scans[scan].pixels])
                if self.gender_checkBox.isChecked():
                    if not 'gender' in self.labels:
                        self.labels.append('gender')
                    self.labels_train.append(self.label_table[self.info_subjects.subjects[subject].gender])

                if self.hand_checkBox.isChecked():
                    if not 'hand' in self.labels:
                        self.labels.append('hand')
                    self.labels_train.append(self.label_table[self.info_subjects.subjects[subject].hand])

                if self.age_checkBox.isChecked():
                    if not 'age' in self.labels:
                        self.labels.append('age')
                    self.labels_train.append(self.info_subjects.subjects[subject].age)

                if self.educ_checkBox.isChecked():
                    if not 'educ' in self.labels:
                        self.labels.append('educ')
                    self.labels_train.append(self.info_subjects.subjects[subject].educ)

                if self.ses_checkBox.isChecked():
                    if not 'ses' in self.labels:
                        self.labels.append('ses')
                    self.labels_train.append(self.info_subjects.subjects[subject].ses)

                if self.mmse_checkBox.isChecked():
                    if not 'mmse' in self.labels:
                        self.labels.append('mmse')
                    self.labels_train.append(self.info_subjects.subjects[subject].mmse)

                if self.cdr_checkBox.isChecked():
                    if not 'cdr' in self.labels:
                        self.labels.append('cdr')
                    self.labels_train.append(self.info_subjects.subjects[subject].cdr)

                if self.e_tiv_checkBox.isChecked():
                    if not 'e_tiv' in self.labels:
                        self.labels.append('e_tiv')
                    self.labels_train.append(self.info_subjects.subjects[subject].e_tiv)

                if self.n_wbv_checkBox.isChecked():
                    if not 'n_wbv' in self.labels:
                        self.labels.append('n_wbv')
                    self.labels_train.append(self.info_subjects.subjects[subject].n_wbv)

                if self.asf_checkBox.isChecked():
                    if not 'asf' in self.labels:
                        self.labels.append('asf')
                    self.labels_train.append(self.info_subjects.subjects[subject].asf)

                if self.delay_checkBox.isChecked():
                    if not 'delay' in self.labels:
                        self.labels.append('delay')
                    self.labels_train.append(self.info_subjects.subjects[subject].delay)

                train += 1
                if train >= subjects_train:
                    break
            if train >= subjects_train:
                    break
        
        self.data_train = np.array(self.data_train).reshape((subjects_train, 256, 256, 1))
        self.data_train = self.data_train.astype('float32') / 255

        self.labels_train = np.array(self.labels_train).reshape((subjects_train, len(self.labels)))

        test = 0
        self.use_test = []

        self.labels_test = []
        self.data_test = []
        subjects_test = int(self.info_subjects.nb_images * self.subjects_test_pour)
        for subject in (subject for subject in self.info_subjects.subjects if subject not in self.use_train):
            self.use_test.append(subject)
            for scan in self.info_subjects.subjects[subject].scans:
                self.data_test.append([self.info_subjects.subjects[subject].scans[scan].pixels])
                if self.gender_checkBox.isChecked():
                    if not 'gender' in self.labels:
                        self.labels.append('gender')
                    self.labels_test.append(self.label_table[self.info_subjects.subjects[subject].gender])

                if self.hand_checkBox.isChecked():
                    if not 'hand' in self.labels:
                        self.labels.append('hand')
                    self.labels_test.append(self.label_table[self.info_subjects.subjects[subject].hand])

                if self.age_checkBox.isChecked():
                    if not 'age' in self.labels:
                        self.labels.append('age')
                    self.labels_test.append(self.info_subjects.subjects[subject].age)

                if self.educ_checkBox.isChecked():
                    if not 'educ' in self.labels:
                        self.labels.append('educ')
                    self.labels_test.append(self.info_subjects.subjects[subject].educ)

                if self.ses_checkBox.isChecked():
                    if not 'ses' in self.labels:
                        self.labels.append('ses')
                    self.labels_test.append(self.info_subjects.subjects[subject].ses)

                if self.mmse_checkBox.isChecked():
                    if not 'mmse' in self.labels:
                        self.labels.append('mmse')
                    self.labels_test.append(self.info_subjects.subjects[subject].mmse)

                if self.cdr_checkBox.isChecked():
                    if not 'cdr' in self.labels:
                        self.labels.append('cdr')
                    self.labels_test.append(self.info_subjects.subjects[subject].cdr)

                if self.e_tiv_checkBox.isChecked():
                    if not 'e_tiv' in self.labels:
                        self.labels.append('e_tiv')
                    self.labels_test.append(self.info_subjects.subjects[subject].e_tiv)

                if self.n_wbv_checkBox.isChecked():
                    if not 'n_wbv' in self.labels:
                        self.labels.append('n_wbv')
                    self.labels_test.append(self.info_subjects.subjects[subject].n_wbv)

                if self.asf_checkBox.isChecked():
                    if not 'asf' in self.labels:
                        self.labels.append('asf')
                    self.labels_test.append(self.info_subjects.subjects[subject].asf)

                if self.delay_checkBox.isChecked():
                    if not 'delay' in self.labels:
                        self.labels.append('delay')
                    self.labels_test.append(self.info_subjects.subjects[subject].delay)

                test += 1
                if test >= subjects_test:
                    break
            if test >= subjects_test:
                    break

        self.data_test = np.array(self.data_test).reshape((subjects_test, 256, 256, 1))
        self.data_test = self.data_test.astype('float32') / 255

        self.labels_test = np.array(self.labels_test).reshape((subjects_test, len(self.labels)))

        self.preprocessing(self.data_train, self.labels_train, self.data_test, self.labels_test)

    def preprocessing(self, data_train, labels_train, data_test, labels_test):
        show_images = ScanShow(self.info_subjects, self.use_train, self.data_train)
        show_images.show()
        self._end_multithread()

    def on_stop(self):
        """ Blabla """
        pass

    def quit(self):
        """ quit the app """
        sys.exit()

    def _start_multithread(self):
        # Setup and start the worker thread to avoid blocking the pyqt GUI
        self.thread = QtCore.QThread()
        self.thread.start()

    def _end_multithread(self):
        self.thread.exit()