# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\sampo\Python\PycharmProjects\SYS818_Project\SYS818\src\main\python\plugins\ScanShow\Ui_ScanShow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ScanShow(object):
    def setupUi(self, ScanShow):
        ScanShow.setObjectName("ScanShow")
        ScanShow.setEnabled(True)
        ScanShow.resize(1596, 1022)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ScanShow.sizePolicy().hasHeightForWidth())
        ScanShow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(ScanShow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setGeometry(QtCore.QRect(20, 30, 1334, 261))
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.layoutWidget = QtWidgets.QWidget(self.widget)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 1332, 709))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_2.setHorizontalSpacing(4)
        self.gridLayout_2.setVerticalSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.back_subject_pushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.back_subject_pushButton.setObjectName("back_subject_pushButton")
        self.gridLayout_2.addWidget(self.back_subject_pushButton, 2, 2, 1, 1)
        self.back_scan_pushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.back_scan_pushButton.setObjectName("back_scan_pushButton")
        self.gridLayout_2.addWidget(self.back_scan_pushButton, 3, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setMinimumSize(QtCore.QSize(50, 0))
        self.label_2.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 2, 0, 1, 1)
        self.next_subject_pushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.next_subject_pushButton.setObjectName("next_subject_pushButton")
        self.gridLayout_2.addWidget(self.next_subject_pushButton, 2, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setMinimumSize(QtCore.QSize(50, 0))
        self.label_4.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 3, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setMinimumSize(QtCore.QSize(0, 15))
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 15))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 5)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 2, 4, 1, 1)
        self.subject_comboBox = QtWidgets.QComboBox(self.layoutWidget)
        self.subject_comboBox.setMinimumSize(QtCore.QSize(300, 0))
        self.subject_comboBox.setMaximumSize(QtCore.QSize(69, 16777215))
        self.subject_comboBox.setObjectName("subject_comboBox")
        self.gridLayout_2.addWidget(self.subject_comboBox, 2, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 3, 4, 1, 1)
        self.scan_comboBox = QtWidgets.QComboBox(self.layoutWidget)
        self.scan_comboBox.setObjectName("scan_comboBox")
        self.gridLayout_2.addWidget(self.scan_comboBox, 3, 1, 1, 1)
        self.next_scan_pushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.next_scan_pushButton.setObjectName("next_scan_pushButton")
        self.gridLayout_2.addWidget(self.next_scan_pushButton, 3, 3, 1, 1)
        self.horizontalLayout_6.addLayout(self.gridLayout_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_6 = QtWidgets.QLabel(self.layoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 1, 5, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.layoutWidget)
        self.label_26.setObjectName("label_26")
        self.gridLayout_3.addWidget(self.label_26, 6, 7, 1, 1)
        self.sid_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.sid_lineEdit.setEnabled(False)
        self.sid_lineEdit.setMinimumSize(QtCore.QSize(250, 0))
        self.sid_lineEdit.setMaximumSize(QtCore.QSize(250, 16777215))
        self.sid_lineEdit.setObjectName("sid_lineEdit")
        self.gridLayout_3.addWidget(self.sid_lineEdit, 1, 2, 1, 1)
        self.scan_type_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.scan_type_lineEdit.setEnabled(False)
        self.scan_type_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.scan_type_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.scan_type_lineEdit.setObjectName("scan_type_lineEdit")
        self.gridLayout_3.addWidget(self.scan_type_lineEdit, 5, 4, 1, 1)
        self.flip_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.flip_lineEdit.setEnabled(False)
        self.flip_lineEdit.setMinimumSize(QtCore.QSize(50, 0))
        self.flip_lineEdit.setMaximumSize(QtCore.QSize(50, 16777215))
        self.flip_lineEdit.setObjectName("flip_lineEdit")
        self.gridLayout_3.addWidget(self.flip_lineEdit, 6, 10, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem2, 2, 13, 1, 1)
        self.ti_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.ti_lineEdit.setEnabled(False)
        self.ti_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.ti_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.ti_lineEdit.setObjectName("ti_lineEdit")
        self.gridLayout_3.addWidget(self.ti_lineEdit, 6, 8, 1, 1)
        self.age_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.age_lineEdit.setEnabled(False)
        self.age_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.age_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.age_lineEdit.setObjectName("age_lineEdit")
        self.gridLayout_3.addWidget(self.age_lineEdit, 1, 8, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.layoutWidget)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 2, 1, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.layoutWidget)
        self.label_21.setObjectName("label_21")
        self.gridLayout_3.addWidget(self.label_21, 5, 5, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.layoutWidget)
        self.label_22.setObjectName("label_22")
        self.gridLayout_3.addWidget(self.label_22, 5, 7, 1, 1)
        self.vox_res_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.vox_res_lineEdit.setEnabled(False)
        self.vox_res_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.vox_res_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.vox_res_lineEdit.setObjectName("vox_res_lineEdit")
        self.gridLayout_3.addWidget(self.vox_res_lineEdit, 5, 6, 1, 1)
        self.orientation_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.orientation_lineEdit.setEnabled(False)
        self.orientation_lineEdit.setMinimumSize(QtCore.QSize(50, 0))
        self.orientation_lineEdit.setMaximumSize(QtCore.QSize(50, 16777215))
        self.orientation_lineEdit.setObjectName("orientation_lineEdit")
        self.gridLayout_3.addWidget(self.orientation_lineEdit, 5, 10, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.layoutWidget)
        self.label_20.setObjectName("label_20")
        self.gridLayout_3.addWidget(self.label_20, 5, 3, 1, 1)
        self.asf_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.asf_lineEdit.setEnabled(False)
        self.asf_lineEdit.setMinimumSize(QtCore.QSize(50, 0))
        self.asf_lineEdit.setMaximumSize(QtCore.QSize(50, 16777215))
        self.asf_lineEdit.setObjectName("asf_lineEdit")
        self.gridLayout_3.addWidget(self.asf_lineEdit, 2, 10, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.layoutWidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout_3.addWidget(self.label_11, 1, 11, 1, 1)
        self.e_tiv_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.e_tiv_lineEdit.setEnabled(False)
        self.e_tiv_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.e_tiv_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.e_tiv_lineEdit.setObjectName("e_tiv_lineEdit")
        self.gridLayout_3.addWidget(self.e_tiv_lineEdit, 2, 6, 1, 1)
        self.hand_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.hand_lineEdit.setEnabled(False)
        self.hand_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.hand_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.hand_lineEdit.setObjectName("hand_lineEdit")
        self.gridLayout_3.addWidget(self.hand_lineEdit, 1, 6, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.layoutWidget)
        self.label_23.setObjectName("label_23")
        self.gridLayout_3.addWidget(self.label_23, 5, 9, 1, 1)
        self.tr_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.tr_lineEdit.setEnabled(False)
        self.tr_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.tr_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.tr_lineEdit.setObjectName("tr_lineEdit")
        self.gridLayout_3.addWidget(self.tr_lineEdit, 6, 4, 1, 1)
        self.cdr_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.cdr_lineEdit.setEnabled(False)
        self.cdr_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.cdr_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.cdr_lineEdit.setObjectName("cdr_lineEdit")
        self.gridLayout_3.addWidget(self.cdr_lineEdit, 2, 4, 1, 1)
        self.ses_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.ses_lineEdit.setEnabled(False)
        self.ses_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.ses_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.ses_lineEdit.setObjectName("ses_lineEdit")
        self.gridLayout_3.addWidget(self.ses_lineEdit, 1, 12, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.layoutWidget)
        self.label_14.setObjectName("label_14")
        self.gridLayout_3.addWidget(self.label_14, 2, 5, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.layoutWidget)
        self.label_16.setObjectName("label_16")
        self.gridLayout_3.addWidget(self.label_16, 2, 7, 1, 1)
        self.scan_number_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.scan_number_lineEdit.setEnabled(False)
        self.scan_number_lineEdit.setMinimumSize(QtCore.QSize(250, 0))
        self.scan_number_lineEdit.setMaximumSize(QtCore.QSize(250, 16777215))
        self.scan_number_lineEdit.setObjectName("scan_number_lineEdit")
        self.gridLayout_3.addWidget(self.scan_number_lineEdit, 5, 2, 1, 1)
        self.n_wbv_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.n_wbv_lineEdit.setEnabled(False)
        self.n_wbv_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.n_wbv_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.n_wbv_lineEdit.setObjectName("n_wbv_lineEdit")
        self.gridLayout_3.addWidget(self.n_wbv_lineEdit, 2, 8, 1, 1)
        self.educ_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.educ_lineEdit.setEnabled(False)
        self.educ_lineEdit.setMinimumSize(QtCore.QSize(50, 0))
        self.educ_lineEdit.setMaximumSize(QtCore.QSize(50, 16777215))
        self.educ_lineEdit.setObjectName("educ_lineEdit")
        self.gridLayout_3.addWidget(self.educ_lineEdit, 1, 10, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.layoutWidget)
        self.label_25.setObjectName("label_25")
        self.gridLayout_3.addWidget(self.label_25, 6, 3, 1, 1)
        self.rect_fov_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.rect_fov_lineEdit.setEnabled(False)
        self.rect_fov_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.rect_fov_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.rect_fov_lineEdit.setObjectName("rect_fov_lineEdit")
        self.gridLayout_3.addWidget(self.rect_fov_lineEdit, 5, 8, 1, 1)
        self.gender_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.gender_lineEdit.setEnabled(False)
        self.gender_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.gender_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.gender_lineEdit.setObjectName("gender_lineEdit")
        self.gridLayout_3.addWidget(self.gender_lineEdit, 1, 4, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.layoutWidget)
        self.label_24.setObjectName("label_24")
        self.gridLayout_3.addWidget(self.label_24, 6, 5, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.layoutWidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 1, 3, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.layoutWidget)
        self.label_19.setObjectName("label_19")
        self.gridLayout_3.addWidget(self.label_19, 5, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.layoutWidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 2, 9, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.layoutWidget)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.gridLayout_3.addWidget(self.label_18, 4, 1, 1, 13)
        self.mmse_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.mmse_lineEdit.setEnabled(False)
        self.mmse_lineEdit.setMinimumSize(QtCore.QSize(250, 0))
        self.mmse_lineEdit.setMaximumSize(QtCore.QSize(250, 16777215))
        self.mmse_lineEdit.setObjectName("mmse_lineEdit")
        self.gridLayout_3.addWidget(self.mmse_lineEdit, 2, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 14)
        self.label_10 = QtWidgets.QLabel(self.layoutWidget)
        self.label_10.setObjectName("label_10")
        self.gridLayout_3.addWidget(self.label_10, 1, 9, 1, 1)
        self.te_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.te_lineEdit.setEnabled(False)
        self.te_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.te_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.te_lineEdit.setObjectName("te_lineEdit")
        self.gridLayout_3.addWidget(self.te_lineEdit, 6, 6, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 1, 1, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.layoutWidget)
        self.label_17.setObjectName("label_17")
        self.gridLayout_3.addWidget(self.label_17, 2, 11, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem3, 1, 13, 1, 1)
        self.delay_lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.delay_lineEdit.setEnabled(False)
        self.delay_lineEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.delay_lineEdit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.delay_lineEdit.setObjectName("delay_lineEdit")
        self.gridLayout_3.addWidget(self.delay_lineEdit, 2, 12, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.layoutWidget)
        self.label_9.setObjectName("label_9")
        self.gridLayout_3.addWidget(self.label_9, 1, 7, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.layoutWidget)
        self.label_15.setObjectName("label_15")
        self.gridLayout_3.addWidget(self.label_15, 2, 3, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.layoutWidget)
        self.label_27.setObjectName("label_27")
        self.gridLayout_3.addWidget(self.label_27, 6, 9, 1, 1)
        self.line = QtWidgets.QFrame(self.layoutWidget)
        self.line.setLineWidth(5)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_3.addWidget(self.line, 3, 1, 1, 12)
        self.verticalLayout_2.addLayout(self.gridLayout_3)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem4)
        self.horizontalLayout_2.addWidget(self.widget)
        self.menubar = QtWidgets.QMenuBar(ScanShow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1400, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.result_layout = QtWidgets.QGraphicsView(ScanShow)
        self.result_layout.setGeometry(QtCore.QRect(340, 300, 991, 711))
        self.result_layout.setObjectName("result_layout")
        self.action_new = QtWidgets.QAction(ScanShow)
        self.action_new.setObjectName("action_new")
        self.action_open = QtWidgets.QAction(ScanShow)
        self.action_open.setObjectName("action_open")
        self.action_save = QtWidgets.QAction(ScanShow)
        self.action_save.setObjectName("action_save")
        self.action_save_as = QtWidgets.QAction(ScanShow)
        self.action_save_as.setObjectName("action_save_as")
        self.action_exit = QtWidgets.QAction(ScanShow)
        self.action_exit.setObjectName("action_exit")
        self.action_presets = QtWidgets.QAction(ScanShow)
        self.action_presets.setObjectName("action_presets")
        self.action_about_scinodes = QtWidgets.QAction(ScanShow)
        self.action_about_scinodes.setObjectName("action_about_scinodes")
        self.action_run = QtWidgets.QAction(ScanShow)
        self.action_run.setObjectName("action_run")
        self.action_stop = QtWidgets.QAction(ScanShow)
        self.action_stop.setObjectName("action_stop")
        self.menuFile.addAction(self.action_new)
        self.menuFile.addAction(self.action_save)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_exit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(ScanShow)
        self.back_subject_pushButton.clicked.connect(ScanShow.on_prev_subject)
        self.next_subject_pushButton.clicked.connect(ScanShow.on_next_subject)
        self.back_scan_pushButton.clicked.connect(ScanShow.on_prev_scan)
        self.next_scan_pushButton.clicked.connect(ScanShow.on_next_scan)
        self.subject_comboBox.currentTextChanged['QString'].connect(ScanShow.on_subject_changed)
        self.scan_comboBox.currentTextChanged['QString'].connect(ScanShow.on_scan_changed)
        QtCore.QMetaObject.connectSlotsByName(ScanShow)

    def retranslateUi(self, ScanShow):
        _translate = QtCore.QCoreApplication.translate
        ScanShow.setWindowTitle(_translate("ScanShow", "Classification algorithm"))
        self.back_subject_pushButton.setText(_translate("ScanShow", "<<"))
        self.back_scan_pushButton.setText(_translate("ScanShow", "<<"))
        self.label_2.setText(_translate("ScanShow", "Subject"))
        self.next_subject_pushButton.setText(_translate("ScanShow", ">>"))
        self.label_4.setText(_translate("ScanShow", "Scan"))
        self.label_3.setText(_translate("ScanShow", "Display image from subjects"))
        self.next_scan_pushButton.setText(_translate("ScanShow", ">>"))
        self.label_6.setText(_translate("ScanShow", "Handedness"))
        self.label_26.setText(_translate("ScanShow", "ti"))
        self.label_13.setText(_translate("ScanShow", "MMSE"))
        self.label_21.setText(_translate("ScanShow", "vox res"))
        self.label_22.setText(_translate("ScanShow", "rect_fov"))
        self.label_20.setText(_translate("ScanShow", "Scan type"))
        self.label_11.setText(_translate("ScanShow", "Socio Economic status"))
        self.label_23.setText(_translate("ScanShow", "orientation"))
        self.label_14.setText(_translate("ScanShow", "eTIV (mm^3)"))
        self.label_16.setText(_translate("ScanShow", "nWBV"))
        self.label_25.setText(_translate("ScanShow", "tr"))
        self.label_24.setText(_translate("ScanShow", "te"))
        self.label_8.setText(_translate("ScanShow", "Gender"))
        self.label_19.setText(_translate("ScanShow", "Scan #"))
        self.label_12.setText(_translate("ScanShow", "asf"))
        self.label_18.setText(_translate("ScanShow", "Scan"))
        self.label_5.setText(_translate("ScanShow", "Subject"))
        self.label_10.setText(_translate("ScanShow", "Education"))
        self.label_7.setText(_translate("ScanShow", "SID"))
        self.label_17.setText(_translate("ScanShow", "delay"))
        self.label_9.setText(_translate("ScanShow", "Age"))
        self.label_15.setText(_translate("ScanShow", "CDR"))
        self.label_27.setText(_translate("ScanShow", "flip"))
        self.menuFile.setTitle(_translate("ScanShow", "&File"))
        self.action_new.setText(_translate("ScanShow", "&New"))
        self.action_new.setShortcut(_translate("ScanShow", "Ctrl+N"))
        self.action_open.setText(_translate("ScanShow", "&Open"))
        self.action_open.setShortcut(_translate("ScanShow", "Ctrl+O"))
        self.action_save.setText(_translate("ScanShow", "&Save"))
        self.action_save.setShortcut(_translate("ScanShow", "Ctrl+S"))
        self.action_save_as.setText(_translate("ScanShow", "Save Pipeline &As"))
        self.action_save_as.setShortcut(_translate("ScanShow", "Ctrl+Shift+S"))
        self.action_exit.setText(_translate("ScanShow", "&Exit"))
        self.action_presets.setText(_translate("ScanShow", "&Presets"))
        self.action_about_scinodes.setText(_translate("ScanShow", "About SciNode"))
        self.action_run.setText(_translate("ScanShow", "Run"))
        self.action_run.setShortcut(_translate("ScanShow", "Ctrl+R"))
        self.action_stop.setText(_translate("ScanShow", "Stop"))