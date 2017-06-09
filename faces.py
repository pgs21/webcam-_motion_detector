import numpy as np
import cv2
import imutils

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
firstFrame = None
face_detected = 0
frame_count = 0

blank_image = np.zeros((480,854,3), np.uint8)

cv2.namedWindow("Frame Delta", cv2.WND_PROP_FULLSCREEN)          
cv2.setWindowProperty("Frame Delta", 0, 1)

while(1):
	ret, img = cap.read()
	img = cv2.flip(img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	grayblur = cv2.GaussianBlur(gray, (21, 21), 0)


	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		face_detected = 5

	if (face_detected >0 ):
		frameDelta = cv2.absdiff(firstFrame, grayblur)
		border=cv2.copyMakeBorder(frameDelta, top=0, bottom=0, left=107, right=107, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
		cv2.imshow("Frame Delta", border)
		face_detected -=1
	k = cv2.waitKey(1)& 0xff
	for dorme in range(0,15):
		cap.read()
	cv2.imshow("Frame Delta", blank_image)
	k = cv2.waitKey(1)& 0xff
	for dorme2 in range(0,1):
		cap.read()

	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break
	elif k==99:
		firstFrame = gray

cap.release()
cv2.destroyAllWindows()