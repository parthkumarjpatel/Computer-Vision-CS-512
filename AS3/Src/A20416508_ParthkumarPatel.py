#Name : Parthkumar Patel
#CWID : A20416508
#SEMESTER : FALL 2018
#COURSE : CS-512 : COMPUTER VISION
#ASSIGNMENT : 3 - Programming Questions

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

#main function which shows the help menu and call all the functions whenever required
def main():
	
	image1, image2, combine_image = getImage() #getting image from user or camera and assignning to varibales
	
	print("Press input key to Process image\n(press 'H' for help, press 'q' to quit):")
	user_input = str(input()) #getting 
	
	while user_input != 'q':
	
		if user_input == 'h': #Estimating image gradients and applying Harris corner detection algorithm
			
			#asking users input for gaussian scale, window size, wieght of the trace, threshold
			n = input("the varience of Gussian scale(n):")
			windowSize = input("windowSize :")
			user_input = input("the weight of the trace in the harris conner detector(user_input)[0, 0.5]:")
			threshold = input("threshold:")

			print("processing harris_corner_detection...")
			image_corner = harris_corner_detection(combine_image, n, windowSize, user_input, threshold) #calling function for harris corner detection
			
			plt.imshow(image_corner, cmap='gray')
			plt.show() #showing processed image
	
		if user_input == 'f':
			
			image_feature = feature_Vector(image1, image2) #calling feature vector function
			plt.imshow(image_feature, cmap='gray')
			plt.show() #showing processed image
	
		if user_input == 'b':
			
			image_betterlocalization = better_Localization(combine_image) #calling better localization function
			plt.imshow(image_betterlocalization, cmap='gray')
			plt.show() #showing processed image
	
		if user_input == 'H':
			print("'h': Estimate image gradients and apply Harris corner detection algorithm.")
			print("'b': Obtain a better localization of each corner.")
			print("'f': Compute a feature vector for each corner were detected.\n")
	
		print("Press input key to Process image\n(press 'H' for help, press 'q' to quit):")
	
		user_input = str(input())

#read image from user or capture from camera
def getImage():
	if len(sys.argv) == 3: #getting input image which was provied by user in command crompt
	
		image1 = cv2.imread(sys.argv[1])
		image2 = cv2.imread(sys.argv[2])
	
	else: #if not provided by user then it will capture image from camera
	
			cap = cv2.VideoCapture(0)
			for i in range(0,15):
				retval1,image1 = cap.read()
				retval2,image2 = cap.read()
	
			if retval1 and retval2:
				cv2.imwrite("capture1.jpg", image1)
				cv2.imwrite("capture2.jpg", image2)
	
	combine_image = np.concatenate((image1, image2), axis=1)
	return image1, image2,combine_image; #returning both individual and combine image

#harris corner detection
def harris_corner_detection(image, n, windowSize, user_input, threshold):
	n = int(n)
	windowSize = int(windowSize)
	user_input = float(user_input)
	threshold = int(threshold)
	copy = image.copy()
	rList = []
	height = image.shape[0]
	width = image.shape[1]	
	offset = int(windowSize / 2)

	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting image to grayscale image
	
	#smoothing the image
	image = np.float32(image)
	kernel = np.ones((n, n), np.float32)/(n * n)
	image = cv2.filter2D(image, -1, kernel)
	
	#calculating image gradient
	dy, dx = np.gradient(image)
	Ixx = dx ** 2
	Ixy = dy * dx
	Iyy = dy ** 2

	#calculating haris corner detection
	for y in range(offset, height - offset):
			for x in range(offset, width - offset):
				windowIxx = Ixx[y - offset : y + offset + 1, x - offset : x + offset + 1]
				windowIxy = Ixy[y - offset : y + offset + 1, x - offset : x + offset + 1]
				windowIyy = Iyy[y - offset : y + offset + 1, x - offset : x + offset + 1]
				Sxx = windowIxx.sum()
				Sxy = windowIxy.sum()
				Syy = windowIyy.sum()
				det = (Sxx * Syy) - (Sxy ** 2)
				trace = Sxx + Syy
				r = det - user_input *(trace ** 2)
				rList.append([x, y, r])
				if r > threshold:
							copy.itemset((y, x, 0), 0)
							copy.itemset((y, x, 1), 0)
							copy.itemset((y, x, 2), 255)
							cv2.rectangle(copy, (x + 10, y + 10), (x - 10, y - 10), (255, 0, 0), 1)
	return copy

#calculating feature vector of each corners were detected
def feature_Vector(image1, image2):

	# Initiate SIFT detector
	orb = cv2.ORB_create()
	
	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(image1,None) # returns keypoints and descriptors
	kp2, des2 = orb.detectAndCompute(image2,None)
	
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	
	# Match descriptors.
	matches = bf.match(des1,des2)
	
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	keypoint1List = []
	keypoint2List = []
	for m in matches:
		(x1, y1) = kp1[m.queryIdx].pt
		(x2, y2) = kp2[m.trainIdx].pt
		keypoint1List.append((x1, y1))
		keypoint2List.append((x2, y2))
	for i in range(0, 50):
		point1 = keypoint1List[i]
		point2 = keypoint2List[i]
		cv2.putText(image1, str(i), (int(point1[0]), int(point1[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
		cv2.putText(image2, str(i), (int(point2[0]), int(point2[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
	image_feature = np.concatenate((image1, image2), axis=1)
	return image_feature

#betterlocalization of image
def better_Localization(image_betterlocalization):
	gray = cv2.cvtColor(image_betterlocalization, cv2.COLOR_BGR2GRAY) #converting image to grayscale image
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04) #performing haris corner detection
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)#applying threshold
	dst = np.uint8(dst)

	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

	result = np.hstack((centroids,corners))
	result = np.int0(result)
	#combining the result into betterlocalization 
	image_betterlocalization[result[:,1],result[:,0]]=[0,0,255]
	image_betterlocalization[result[:,3],result[:,2]] = [0,255,0]
	return image_betterlocalization

if __name__ == '__main__':
	main()
