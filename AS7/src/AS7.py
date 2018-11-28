#Name : Parthkumar Patel
#CWID : A20416508
#SEMESTER : FALL 2018
#COURSE : CS-512 : COMPUTER VISION
#Assignment : 7 - Optical Flow Estimation

'''
Optical Flow Estimation
Help: 
    e.g. .\AS6.py ..\data\Walk2.mpg
    Press p :  pause or release current image
    Press ESC : to exit 
'''
from scipy import ndimage
import numpy as np
import cv2
import sys

"""
image1: image at t=0
image2: image at t=1
alpha: regularization constant
iteration: number of iteration
"""
def calculateFlowUsingHornSchunk(image1, image2, alpha, iteration):

	image1 = image1.astype(np.float32) #Copy of the image 1 and cast to a specified type.
	image2 = image2.astype(np.float32) #Copy of the image 2 and cast to a specified type.

	#set up initial velocities
	Intial_U = np.zeros([image1.shape[0],image1.shape[1]]) 
	Intial_V = np.zeros([image1.shape[0],image1.shape[1]])

	fx = cv2.Sobel(image1,cv2.CV_32F,1,0,ksize=1) # Calculates the first order derivatives of X using an extended Sobel operator
	fy = cv2.Sobel(image1,cv2.CV_32F,0,1,ksize=1) # Calculates the first order derivatives of Y using an extended Sobel operator.
	ft = image2 - image1
	i = 0

	#function to reduce the error
	for i in range(iteration):

		#Compute local averages of the flow vectors
		#inputs are Intial flow vector, Horn Schunk Kernel value same dimension as intial flow vector, mode is Constant so the input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
		Averagr_U = ndimage.convolve(Intial_U, HornSchunk_Kernel, mode = 'constant', cval = 0.0)
		Averagr_V = ndimage.convolve(Intial_V, HornSchunk_Kernel, mode = 'constant', cval = 0.0)
		
		#common part of update step
		derivative = (fx*Averagr_U + fy*Averagr_V + ft) / (alpha**2 + fx**2 + fy**2)
		
		# iterative step
		Intial_U = Averagr_U - fx * derivative
		Intial_V = Averagr_V - fy * derivative
		
		i += 1
		
	flow = np.dstack((Intial_U,Intial_V)) #Stack arrays in sequence depth wis
	return flow #returning flow
"""
image : current image
flow : optical flow of current image
"""
def ShowFlowOfCurrentImage(image, flow, step=16):
    height, weight = image.shape[:2] #getting height and weight of image
    y, x = np.mgrid[step/2:height:step, step/2:weight:step].reshape(2,-1).astype(int) #calculating x and y using the step,weight and weight and reshaping it as type interger
    fx, fy = flow[y,x].T #defining fx and fy 
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2) # Stack arrays in sequence vertically (row wise).
    lines = np.int32(lines + 0.5)
    visulization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # converting gray scale image to color
    cv2.polylines(visulization, lines, 0, (0, 255, 0)) #Draws several polygonal curves
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(visulization, (x1, y1), 1, (0, 255, 0), -1) #Draws a circle.
    return visulization #returning a visulization

print(__doc__)	# to print the help message

cap = cv2.VideoCapture(sys.argv[1])#reading video file argument from user
ret, frame = cap.read() #returns a bool value of ret. If frame is read correctly, it will be True. Frame returns the matrix of frmae
frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts the frame into gray scale 
frame_grayscale = cv2.GaussianBlur(frame_grayscale,(9,9),2) #Blurs an frame using a Gaussian filter.

HornSchunk_Kernel =np.array([[1/12, 1/6, 1/12], [1/6, 0, 1/6], [1/12, 1/6, 1/12]],float)# declaring Horn Schink Kernal

#continiously reading a frame from video file
while(cap.isOpened()):
	ret, image = cap.read() #reading a frame
	if ret: 
		ch = cv2.waitKey(30)
		if ch == 27: #if press ESC program will end
			break
		if ch == ord('p') or ch == ord('P'): # if p is pressed then it will calculate the flow of image and video will pause
			print("Press 'p' to pause or release current image")
			
			image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting current image to grayscale
			image_grayscale = cv2.GaussianBlur(image_grayscale,(9,9),2) #blurs an image using a guassian filter
			
			flow = 2 * calculateFlowUsingHornSchunk(frame_grayscale, image_grayscale, 0.001, 8)  #Calculates the optical flow for two images using Horn-Schunck algorithm.
			frame_grayscale = image_grayscale #assigning a current frame as a previous frame 
			cv2.imshow('Optical flow', ShowFlowOfCurrentImage(image_grayscale, flow)) #show the flow
			#it will wait for user's input
			while True:
				ch2 = cv2.waitKey(1)
				if ch2 == (ord('p') or ch == ord('P')): #video will start playing 
					cv2.destroyAllWindows()
					break
				elif ch2 == 27: # program will end
					break
		cv2.imshow('Video', image) #showing video

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
