#Name : Parthkumar Patel
#CWID : A20416508
#SEMESTER : FALL 2018
#COURSE : CS-512 : COMPUTER VISION
#ASSIGNMENT : 5 - Programming Questions
'''
Extract feature points using the openCV functions
-------------------------
Usage:
    extractfeats.py 
-------------------------
Keys:
    select image window
    press any key to exit
-------------------------   
Output:
    correspondencePoints.txt
    A point correspondence file (3D-2D)
'''
import cv2
import numpy as np
import sys

print(__doc__)
image = cv2.imread('C:/Users/Parth/Desktop/IITC/CV/AS5/assignment_5/chessboard.jpg') #image to be read from folder 

# termination criteria
termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
object_points = np.zeros((6*7,3), np.float32) 
object_points[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#converting image to grayscale

#Finds the positions of internal corners of the chessboard.
ret, corners = cv2.findChessboardCorners(grayscale_image, (7,6), None)

# If corners are found then it will proceeds further
if ret:
	#corner subpix refines the corner locations
	refineCorners=cv2.cornerSubPix(grayscale_image,corners, (11,11), (-1,-1), termination_criteria) #process of corner position refinement stops either after criteria.maxCount iterations or when the corner position moves by less than criteria.epsilon on some iteration.

	# Draw and display the corners
	cv2.drawChessboardCorners(image, (7,6), refineCorners, ret)
	
	#displaying the image with corners
	cv2.imshow("Chess Board Corners", image)
	
	#opening a file to write correspondence points
	file = open("correspondencePoints.txt", "w")
	for objp, cornerposition in zip(object_points, corners.reshape(-1,2)): #zip() function returns an iterator of tuples based on the iterable object
		file.write(str(objp[0]) + ' ' + str(objp[1]) + ' ' + str(objp[2]) + ' ' + str(cornerposition[0]) + ' ' + str(cornerposition[1]) + '\n')
	file.close()

	cv2.waitKey(0)
cv2.destroyAllWindows()




