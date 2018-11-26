import face_recognition #Import modules
import numpy as np
import cv2

freddie = cv2.imread('freddiem.png',-1) #Reads face filter
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


class Mask_ops:

	def overlay(self, frame, freddie, pos=(0,0), scale = 1): #Function for overlaying filter

		freddie = cv2.resize(freddie, (0,0), fx=scale, fy=scale)
		h, w, _ = freddie.shape
		rows, cols, _ = frame.shape
		y, x = pos[0], pos[1]

		for i in range(h):
			for j in range(w):
				if x + i >= rows or y + j >= cols:
					continue

				alpha = float(freddie[i][j][3] / 255.0)
				frame[x + i][y + j] = alpha * freddie[i][j][:3] + (1 - alpha) * frame[x + i][y + j]

		return frame

	


while "Freddie" == "Freddie": #True

	freddie_ops = Mask_ops()

	ret, image = cap.read()
	
	image_frame = image[:, :, ::-1]
	det = face_recognition.face_locations(image_frame) #Detects faces
	faces = [(0,0,0,0)] #Initialize set of faces

	if det != []: #If faces are detected

		faces = [[det[0][3], det[0][0], abs(det[0][3] - det[0][1]) + 150, abs(det[0][0] - det[0][2])]] #Points in face
		
		
		for (x, y, w, h) in faces: #Coordinates of faces
			x -= 65 #Manually adjust filter placement
			w -= 35
			y -= 50
			h -= 10
		

			freddie_ymin = int(y - 3 * h / 5) #Equation to solve locations
			freddie_ymax = int(y + 8 * h / 5)

			sh_freddie = freddie_ymax - freddie_ymin


			face_frame = image[freddie_ymin:freddie_ymax, x:x + w]
			freddie_resized = cv2.resize(freddie, (w, sh_freddie), interpolation=cv2.INTER_CUBIC) #Resize face filter to fit face
			
			
			freddie_ops.overlay(face_frame, freddie_resized) #Call overlay function


	cv2.imshow('Ready, Freddie?', image) #Show video stream


	if cv2.waitKey(30) & 0xFF == ord("q"): #"q" to quit
		break

cap.release()
cv2.destroyAllWindows()
#Merry Christmas!
