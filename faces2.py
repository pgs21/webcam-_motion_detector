import numpy as np
import cv2
import imutils

import subprocess
import os.path

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)

firstFrame = None
firstFrameleft = None
firstFrameright = None

face_detected = 0
frame_count = 0
playing = 0
sair = 1

blank_image = np.zeros((480,854,3), np.uint8)

cv2.namedWindow("Frame Delta", cv2.WND_PROP_FULLSCREEN)          
cv2.setWindowProperty("Frame Delta", 0, 1)

f = 'Porches_and_Universes.mp3'
if os.path.isfile(f):
	subprocess.Popen(['audacious',f])
	subprocess.Popen(['audtool','--playback-stop'])


def imgprocess (capture):
	capture.grab()
	capture.grab()
	capture.grab()
	ret,img = capture.read()
	
	if not ret :
		print "cameras failed"
	img = cv2.flip(img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = imutils.resize(gray, width=500)
	
	
	return gray

def grayprocess(gray):
	grayblur = cv2.GaussianBlur(gray, (21, 21), 0)
	global firstFrame 
	if firstFrame is None:
		firstFrame = grayblur
	frameDelta = cv2.absdiff(firstFrame, grayblur)
	border=cv2.copyMakeBorder(frameDelta, top=0, bottom=0, left=107, right=107, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	return border

def grayprocessleft(gray):
	grayblur = cv2.GaussianBlur(gray, (21, 21), 0)
	global firstFrameleft
	if firstFrameleft is None:
		firstFrameleft = grayblur
	frameDelta = cv2.absdiff(firstFrameleft, grayblur)
	border=cv2.copyMakeBorder(frameDelta, top=0, bottom=0, left=107, right=107, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	return border

def grayprocessright(gray):
	grayblur = cv2.GaussianBlur(gray, (21, 21), 0)
	global firstFrameright
	if firstFrameright is None:
		firstFrameright = grayblur
	frameDelta = cv2.absdiff(firstFrameright, grayblur)
	border=cv2.copyMakeBorder(frameDelta, top=0, bottom=0, left=107, right=107, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	return border

def faceprocess(gray):
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	face = 0
	for (x,y,w,h) in faces:
		face=  5
		global playing
		if playing == 0:
			subprocess.Popen(['audtool','--playback-play'])
			playing = 1

	return face

def blink(cap):
	k = cv2.waitKey(1)& 0xff
	if k == 27:
		return 1
	for dorme in range(0,15):
		cap.grab()
	cv2.imshow("Frame Delta", blank_image)
	k = cv2.waitKey(1)& 0xff
	if k == 27:
		return 1
	for dorme2 in range(0,1):
		cap.grab()
	return 0

cv2.imshow("Frame Delta", blank_image)

while(sair):
	
	cap2 = cv2.VideoCapture(4)
	cap3 = cv2.VideoCapture(5)
	face = 0
	for cinco in range(0, 6):
		gray = imgprocess(cap)
		face = faceprocess(gray)
		border = grayprocess(gray)
		
		if (face > 0 ):
			cv2.imshow("Frame Delta", border)
			face -=1
			if blink(cap):
				sair = 0

	graymain = imgprocess(cap)
	face = faceprocess(graymain)

	for cinco2 in range(0, 6):
		graymain = imgprocess(cap)
		face = faceprocess(graymain)

		gray = imgprocess(cap2)
		border = grayprocessleft(gray)
		if (face >0 ):
			cv2.imshow("Frame Delta", border)
			face -=1
			if blink(cap2):
				sair = 0

	cap2.release()
	
	

	for cinco3 in range(0, 6):
		graymain = imgprocess(cap)
		face = faceprocess(graymain)

		gray = imgprocess(cap3)
		border = grayprocessright(gray)
		if (face >0 ):
			cv2.imshow("Frame Delta", border)
			face -=1
			if blink(cap3):
				sair = 0

	cap3.release()
	
	
	if(face_detected ==0) and (playing == 1):
		subprocess.Popen(['audtool','--playback-stop'])
		playing = 0
	








	

subprocess.Popen(['audtool','--shutdown'])

cap.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()