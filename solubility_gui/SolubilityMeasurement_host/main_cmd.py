# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_main_window(object):
    def setupUi(self, main_window):
        main_window.setObjectName("main_window")
        main_window.setWindowModality(QtCore.Qt.NonModal)
        main_window.resize(1000, 965)
        main_window.setMinimumSize(QtCore.QSize(1000, 965))
        main_window.setMaximumSize(QtCore.QSize(1000, 965))
        self.central_widget = QtWidgets.QWidget(main_window)
        self.central_widget.setObjectName("central_widget")
        self.front_camera_view = QtWidgets.QGraphicsView(self.central_widget)
        self.front_camera_view.setGeometry(QtCore.QRect(10, 10, 384, 250))
        self.front_camera_view.setObjectName("front_camera_view")
        self.display_camera_view = QtWidgets.QGraphicsView(self.central_widget)
        self.display_camera_view.setGeometry(QtCore.QRect(406, 10, 384, 250))
        self.display_camera_view.setAlignment(QtCore.Qt.AlignCenter)
        self.display_camera_view.setObjectName("display_camera_view")
        self.front_camera_label = QtWidgets.QLabel(self.central_widget)
        self.front_camera_label.setGeometry(QtCore.QRect(90, 275, 101, 16))
        self.front_camera_label.setObjectName("front_camera_label")
        self.rear_camera_label = QtWidgets.QLabel(self.central_widget)
        self.rear_camera_label.setGeometry(QtCore.QRect(96, 885, 91, 16))
        self.rear_camera_label.setObjectName("rear_camera_label")
        self.rear_camera_view = QtWidgets.QGraphicsView(self.central_widget)
        self.rear_camera_view.setGeometry(QtCore.QRect(16, 300, 768, 576))
        self.rear_camera_view.setObjectName("rear_camera_view")
        self.front_camera_capture_button = QtWidgets.QPushButton(self.central_widget)
        self.front_camera_capture_button.setGeometry(QtCore.QRect(10, 270, 75, 23))
        self.front_camera_capture_button.setObjectName("front_camera_capture_button")
        self.rear_camera_capture_button = QtWidgets.QPushButton(self.central_widget)
        self.rear_camera_capture_button.setGeometry(QtCore.QRect(10, 880, 75, 23))
        self.rear_camera_capture_button.setObjectName("rear_camera_capture_button")
        self.display_camera_capture_button = QtWidgets.QPushButton(self.central_widget)
        self.display_camera_capture_button.setGeometry(QtCore.QRect(406, 270, 75, 23))
        self.display_camera_capture_button.setObjectName("display_camera_capture_button")
        self.display_camera_label = QtWidgets.QLabel(self.central_widget)
        self.display_camera_label.setGeometry(QtCore.QRect(486, 275, 231, 16))
        self.display_camera_label.setObjectName("display_camera_label")
        self.client_state_group_box = QtWidgets.QGroupBox(self.central_widget)
        self.client_state_group_box.setGeometry(QtCore.QRect(630, 880, 161, 81))
        self.client_state_group_box.setObjectName("client_state_group_box")
        self.camera_state_label = QtWidgets.QLabel(self.client_state_group_box)
        self.camera_state_label.setGeometry(QtCore.QRect(30, 30, 81, 16))
        self.camera_state_label.setObjectName("camera_state_label")
        self.display_state_view = QtWidgets.QWidget(self.client_state_group_box)
        self.display_state_view.setGeometry(QtCore.QRect(10, 60, 16, 16))
        self.display_state_view.setObjectName("display_state_view")
        self.camera_state_view = QtWidgets.QWidget(self.client_state_group_box)
        self.camera_state_view.setGeometry(QtCore.QRect(10, 30, 16, 16))
        self.camera_state_view.setObjectName("camera_state_view")
        self.display_state_label = QtWidgets.QLabel(self.client_state_group_box)
        self.display_state_label.setGeometry(QtCore.QRect(30, 60, 81, 16))
        self.display_state_label.setObjectName("display_state_label")
        self.image_path_label = QtWidgets.QLabel(self.central_widget)
        self.image_path_label.setGeometry(QtCore.QRect(90, 915, 511, 16))
        self.image_path_label.setObjectName("image_path_label")
        self.send_image_to_display_button = QtWidgets.QPushButton(self.central_widget)
        self.send_image_to_display_button.setGeometry(QtCore.QRect(10, 910, 75, 23))
        self.send_image_to_display_button.setObjectName("send_image_to_display_button")
        self.calibration_ck_box = QtWidgets.QCheckBox(self.central_widget)
        self.calibration_ck_box.setGeometry(QtCore.QRect(360, 915, 150, 16))
        self.calibration_ck_box.setObjectName("calibration_ck_box")
        self.bg_display_button = QtWidgets.QPushButton(self.central_widget)
        self.bg_display_button.setGeometry(QtCore.QRect(250, 940, 91, 23))
        self.bg_display_button.setObjectName("bg_display_button")
        self.save_path_label = QtWidgets.QLabel(self.central_widget)
        self.save_path_label.setGeometry(QtCore.QRect(10, 940, 231, 16))
        self.save_path_label.setObjectName("save_path_label")
        self.save_filename_lineedit = QtWidgets.QLineEdit(self.central_widget)
        self.save_filename_lineedit.setGeometry(QtCore.QRect(350, 940, 271, 20))
        self.save_filename_lineedit.setObjectName("save_filename_lineedit")
        self.torch_toggle_button = QtWidgets.QPushButton(self.central_widget)
        self.torch_toggle_button.setGeometry(QtCore.QRect(550, 880, 75, 23))
        self.torch_toggle_button.setObjectName("torch_toggle_button")
        self.Feature_groupbox = QtWidgets.QGroupBox(self.central_widget)
        self.Feature_groupbox.setGeometry(QtCore.QRect(790, 600, 201, 361))
        self.Feature_groupbox.setObjectName("Feature_groupbox")
        self.supp_ratio_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.supp_ratio_label.setGeometry(QtCore.QRect(10, 20, 71, 16))
        self.supp_ratio_label.setObjectName("supp_ratio_label")
        self.mmg_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.mmg_label.setGeometry(QtCore.QRect(10, 45, 71, 16))
        self.mmg_label.setObjectName("mmg_label")
        self.msg_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.msg_label.setGeometry(QtCore.QRect(10, 70, 64, 15))
        self.msg_label.setObjectName("msg_label")
        self.smg_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.smg_label.setGeometry(QtCore.QRect(10, 95, 64, 15))
        self.smg_label.setObjectName("smg_label")
        self.ssg_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.ssg_label.setGeometry(QtCore.QRect(10, 120, 64, 15))
        self.ssg_label.setObjectName("ssg_label")
        self.curve_a_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.curve_a_label.setGeometry(QtCore.QRect(10, 145, 64, 15))
        self.curve_a_label.setObjectName("curve_a_label")
        self.curve_b_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.curve_b_label.setGeometry(QtCore.QRect(10, 170, 64, 15))
        self.curve_b_label.setObjectName("curve_b_label")
        self.curve_c_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.curve_c_label.setGeometry(QtCore.QRect(10, 195, 64, 15))
        self.curve_c_label.setObjectName("curve_c_label")
        self.std_range_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.std_range_label.setGeometry(QtCore.QRect(10, 220, 64, 15))
        self.std_range_label.setObjectName("std_range_label")
        self.gradient_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.gradient_label.setGeometry(QtCore.QRect(10, 245, 64, 15))
        self.gradient_label.setObjectName("gradient_label")
        self.output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.output_label.setGeometry(QtCore.QRect(10, 340, 64, 15))
        self.output_label.setObjectName("output_label")
        self.supp_ratio_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.supp_ratio_output_label.setGeometry(QtCore.QRect(100, 20, 91, 16))
        self.supp_ratio_output_label.setObjectName("supp_ratio_output_label")
        self.mmg_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.mmg_output_label.setGeometry(QtCore.QRect(100, 45, 91, 16))
        self.mmg_output_label.setObjectName("mmg_output_label")
        self.msg_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.msg_output_label.setGeometry(QtCore.QRect(100, 70, 91, 16))
        self.msg_output_label.setObjectName("msg_output_label")
        self.smg_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.smg_output_label.setGeometry(QtCore.QRect(100, 95, 91, 16))
        self.smg_output_label.setObjectName("smg_output_label")
        self.ssg_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.ssg_output_label.setGeometry(QtCore.QRect(100, 120, 91, 16))
        self.ssg_output_label.setObjectName("ssg_output_label")
        self.curve_a_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.curve_a_output_label.setGeometry(QtCore.QRect(100, 145, 91, 16))
        self.curve_a_output_label.setObjectName("curve_a_output_label")
        self.curve_b_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.curve_b_output_label.setGeometry(QtCore.QRect(100, 170, 91, 16))
        self.curve_b_output_label.setObjectName("curve_b_output_label")
        self.curve_c_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.curve_c_output_label.setGeometry(QtCore.QRect(100, 195, 91, 16))
        self.curve_c_output_label.setObjectName("curve_c_output_label")
        self.std_range_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.std_range_output_label.setGeometry(QtCore.QRect(100, 220, 91, 16))
        self.std_range_output_label.setObjectName("std_range_output_label")
        self.gradient_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.gradient_output_label.setGeometry(QtCore.QRect(100, 245, 91, 16))
        self.gradient_output_label.setObjectName("gradient_output_label")
        self.output_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.output_output_label.setGeometry(QtCore.QRect(100, 340, 91, 16))
        self.output_output_label.setObjectName("output_output_label")
        self.particle_sum_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.particle_sum_label.setGeometry(QtCore.QRect(10, 270, 81, 16))
        self.particle_sum_label.setObjectName("particle_sum_label")
        self.particle_sum_output_label = QtWidgets.QLabel(self.Feature_groupbox)
        self.particle_sum_output_label.setGeometry(QtCore.QRect(100, 270, 91, 16))
        self.particle_sum_output_label.setObjectName("particle_sum_output_label")
        self.processing_button = QtWidgets.QPushButton(self.central_widget)
        self.processing_button.setGeometry(QtCore.QRect(790, 570, 101, 23))
        self.processing_button.setObjectName("processing_button")
        self.bg_select_combobox = QtWidgets.QComboBox(self.central_widget)
        self.bg_select_combobox.setGeometry(QtCore.QRect(790, 540, 101, 22))
        self.bg_select_combobox.setObjectName("bg_select_combobox")
        self.bg_select_combobox.addItem("")
        self.bg_select_combobox.addItem("")
        self.save_button = QtWidgets.QPushButton(self.central_widget)
        self.save_button.setGeometry(QtCore.QRect(550, 910, 75, 23))
        self.save_button.setObjectName("save_button")
        main_window.setCentralWidget(self.central_widget)

        self.retranslateUi(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslateUi(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "Solubility Measurement"))
        self.front_camera_label.setText(_translate("main_window", "Front Camera"))
        self.rear_camera_label.setText(_translate("main_window", "Rear Camera"))
        self.front_camera_capture_button.setText(_translate("main_window", "Capture"))
        self.rear_camera_capture_button.setText(_translate("main_window", "Capture"))
        self.display_camera_capture_button.setText(_translate("main_window", "Capture"))
        self.display_camera_label.setText(_translate("main_window", "Display Camera"))
        self.client_state_group_box.setTitle(_translate("main_window", "Clients"))
        self.camera_state_label.setText(_translate("main_window", "Camera"))
        self.display_state_label.setText(_translate("main_window", "Display"))
        self.image_path_label.setText(_translate("main_window", "(Double click here to browse an image.)"))
        self.send_image_to_display_button.setText(_translate("main_window", "Display"))
        self.calibration_ck_box.setText(_translate("main_window", "Calibration Mode"))
        self.bg_display_button.setText(_translate("main_window", "BG Display"))
        self.save_path_label.setText(_translate("main_window", "(Double click here to set up path.)"))
        self.save_filename_lineedit.setText(_translate("main_window", "(Save file name)"))
        self.torch_toggle_button.setText(_translate("main_window", "Torch"))
        self.Feature_groupbox.setTitle(_translate("main_window", "Feature/Output"))
        self.supp_ratio_label.setText(_translate("main_window", "supp_ratio"))
        self.mmg_label.setText(_translate("main_window", "mmg"))
        self.msg_label.setText(_translate("main_window", "msg"))
        self.smg_label.setText(_translate("main_window", "smg"))
        self.ssg_label.setText(_translate("main_window", "ssg"))
        self.curve_a_label.setText(_translate("main_window", "curve_a"))
        self.curve_b_label.setText(_translate("main_window", "curve_b"))
        self.curve_c_label.setText(_translate("main_window", "curve_c"))
        self.std_range_label.setText(_translate("main_window", "std_range"))
        self.gradient_label.setText(_translate("main_window", "gradient"))
        self.output_label.setText(_translate("main_window", "output"))
        self.supp_ratio_output_label.setText(_translate("main_window", "0"))
        self.mmg_output_label.setText(_translate("main_window", "0"))
        self.msg_output_label.setText(_translate("main_window", "0"))
        self.smg_output_label.setText(_translate("main_window", "0"))
        self.ssg_output_label.setText(_translate("main_window", "0"))
        self.curve_a_output_label.setText(_translate("main_window", "0"))
        self.curve_b_output_label.setText(_translate("main_window", "0"))
        self.curve_c_output_label.setText(_translate("main_window", "0"))
        self.std_range_output_label.setText(_translate("main_window", "0"))
        self.gradient_output_label.setText(_translate("main_window", "0"))
        self.output_output_label.setText(_translate("main_window", "-"))
        self.particle_sum_label.setText(_translate("main_window", "particle_sum"))
        self.particle_sum_output_label.setText(_translate("main_window", "0"))
        self.processing_button.setText(_translate("main_window", "Processing"))
        self.bg_select_combobox.setItemText(0, _translate("main_window", "White"))
        self.bg_select_combobox.setItemText(1, _translate("main_window", "Check_pattern"))
        self.save_button.setText(_translate("main_window", "Save"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = Ui_main_window()
    ui.setupUi(main_window)
    main_window.show()
    sys.exit(app.exec_())

