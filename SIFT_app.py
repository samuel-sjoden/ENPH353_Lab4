#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np



class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)
		self._cam_id = 0
		self._cam_fps = 60
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.refrence_image = None
		self.reference_kp = None
		self.reference_des = None

		self.MIN_MATCH_COUNT = 10

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		self._camera_device = cv2.VideoCapture(self._cam_id)
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1000 / self._cam_fps)

	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)
		self.refrence_image = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		sift = cv2.SIFT_create()
		self.reference_kp, self.reference_des = sift.detectAndCompute(self.refrence_image,None)
		print("Loaded template image file: " + self.template_path)

		# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):
		ret, frame = self._camera_device.read()
		#TODO run SIFT on the captured frame
		if self.refrence_image is not None:
			# Identify keypoints with sift
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			sift = cv2.SIFT_create()
			kp, des = sift.detectAndCompute(gray,None)

			# Match descriptor vectors with FLANN based matching
			matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
			knn_matches = matcher.knnMatch(self.reference_des, des, 2)

			# Filter matches with Lowes ratio test
			ratio_thresh = 0.7
			good_matches = []
			for m,n in knn_matches:
				if m.distance < ratio_thresh * n.distance:
					good_matches.append(m)

			#Obtained from https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
			if len(good_matches)>self.MIN_MATCH_COUNT:
				src_pts = np.float32([ self.reference_kp[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
				dst_pts = np.float32([ kp[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
				M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
				h,w = self.refrence_image.shape[:2]
				pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
				dst = cv2.perspectiveTransform(pts,M)
				frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
			else:
				print( "Not enough matches are found - {}/{}".format(len(good_matches), self.MIN_MATCH_COUNT) )
				img_matches = np.empty((max(frame.shape[0], self.refrence_image.shape[0]), (frame.shape[1]+self.refrence_image.shape[1]), 3), dtype=np.uint8)
				cv2.drawMatches(self.refrence_image, self.reference_kp, frame, kp, good_matches, img_matches, flags=cv2.DrawMatchesFlags_DEFAULT)
				frame = img_matches
		else:
			print("No reference image provided!")



		pixmap = self.convert_cv_to_pixmap(frame)
		self.live_image_label.setPixmap(pixmap)

	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())

