#Name : Parthkumar Patel, Arpit Hasmukhbhai Patel
#CWID : A20416508, A20424085
#SEMESTER : FALL 2018
#COURSE : CS-512 : COMPUTER VISION
#Project : Panoramic Image Stitching


import sys
import cv2
import math
import numpy as np
import multiprocessing as multiprocess
import featureextractionandmatching as featureextractionandmatching
import imagestitching
import constant as const
import os
import matplotlib.pyplot as plt

"""
load images and focal length from the input directory.

Args:
sourceDirectory: the directory that have set of input images and a 'image_list.txt'

Returns:
An array of images and its focalLength
"""
def load_images_and_focal_length(sourceDirectory):
	filenames = [] #array to store the filenames
	focalLength = [] #array to store focal length of images
	file = open(os.path.join(sourceDirectory, 'image_list.txt')) #open image list file
	#read filename and its focallength from file
	for line in file:
		if (line[0] == '#'):
			continue
		(filename, file, *rest) = line.split()
		filenames += [filename]
		focalLength += [float(file)]

	imageList = [cv2.imread(os.path.join(sourceDirectory, file), 1) for file in filenames] #making an array of list of images

	return (imageList, focalLength) #return an array of images and its focal length


"""
Project image to cylinder

Args:
image: input image
focalLength: input image's focal length

Return:
Cylindrical projection of input image
"""
def image_cylindrical_projection(image, focalLength):
	height, width, _ = image.shape
	cylinder_projection = np.zeros(shape=image.shape, dtype=np.uint8)

	for y in range(-int(height/2), int(height/2)):
		for x in range(-int(width/2), int(width/2)):
			cylinder_x = focalLength*math.atan(x/focalLength) 
			cylinder_y = focalLength*y/math.sqrt(x**2+focalLength**2)
			
			cylinder_x = round(cylinder_x + width/2)
			cylinder_y = round(cylinder_y + height/2)

			if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
				cylinder_projection[cylinder_y][cylinder_x] = image[y+int(height/2)][x+int(width/2)]

	# cropImage black border
	_, thresh = cv2.threshold(cv2.cvtColor(cylinder_projection, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY) #Applies a fixed-level threshold to each array element
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Finds contours in a binary image
	x, y, w, h = cv2.boundingRect(contours[0]) #Calculates the up-right bounding rectangle of a point set.
		
	return cylinder_projection[y:y+h, x:x+w] #return cylindrical projection of an input image

"""
Plot matched pair of images

Args:
p1 : input image
p2 : input image
matchedpair : matched pair of images
"""
def plotMatchedPairs(p1, p2, matchedpair):
	_, offset, _ = p1.shape
	plt_image = np.concatenate((p1, p2), axis=1)
	plt.figure(figsize=(10,10))
	plt.imshow(plt_image)
	for i in range(len(matchedpair)):
		plt.scatter(x=matchedpair[i][0][1], y=matchedpair[i][0][0], c='r')
		plt.plot([matchedpair[i][0][1], offset+matchedpair[i][1][1]], [matchedpair[i][0][0], matchedpair[i][1][0]], 'y-', lw=1)
		plt.scatter(x=offset+matchedpair[i][1][1], y=matchedpair[i][1][0], c='b')
	plt.show()
	cv2.waitKey(0)

"""
main function

"""
if __name__ == '__main__':

	#if system argument is not 2 then shows it help menu how to run 
	if len(sys.argv) != 2:
		print('[Usage] python scriptname input image directory')
		print('[Exampe] python scriptname.py ..\input_image\Book')
		sys.exit(0) #exit from program 

	input_directoryName = sys.argv[1] #takes first argument as input directory name

	pool = multiprocess.Pool(multiprocess.cpu_count()) #getting number of CPU

	imageList, focalLength = load_images_and_focal_length(input_directoryName) #calling load images and focal length function 

	print('Warp images to cylinder')
	cylinder_image_list = pool.starmap(image_cylindrical_projection, [(imageList[i], focalLength[i]) for i in range(len(imageList))]) #pass all images to image cylindrical projection function


	_, imageWidth, _ = imageList[0].shape #getting shape of first image
	stitchedImages = cylinder_image_list[0].copy() #copying first cylinder image as stitched images

	shifts = [[0, 0]] #declaring shifts as [0 0]
	store_feature = [[], []] #array to store feature of previous images

	# add first image for end to end align
	#cylinder_image_list += [stitchedImages]

	#this for loop will take all cylinder images and extract features and matches every pair of images and stitch it together 
	for i in range(1, len(cylinder_image_list)):
		print('Computing .... '+str(i+1)+'/'+str(len(cylinder_image_list)))

		image1 = cylinder_image_list[i-1] #take first image
		image2 = cylinder_image_list[i]	#take second image

		print(' - Previous image .... ', end='', flush=True)
		descriptors1, position1 = store_feature #taking descriptor and position from an array
		
		#for the first image
		if len(descriptors1) == 0: 
			corner_response1 = featureextractionandmatching.harrisCornerDetection(image1, pool) #applying harris corner detection on image 1
			descriptors1, position1 = featureextractionandmatching.extractDescofCornerResponse(image1, corner_response1, kernel=const.DESCRIPTOR_SIZE, threshold=const.FEATURE_THRESHOLD) #applying extract description from corner response image
		print(str(len(descriptors1))+' features extracted.') #print extracted feature from image 1

		print(' - Image '+str(i+1)+' .... ', end='', flush=True) #find feature in second image
		corner_response2 = featureextractionandmatching.harrisCornerDetection(image2, pool) #applying harris corner detection on image 1
		descriptors2, position2 = featureextractionandmatching.extractDescofCornerResponse(image2, corner_response2, kernel=const.DESCRIPTOR_SIZE, threshold=const.FEATURE_THRESHOLD)#applying extract description from corner response image
		print(str(len(descriptors2))+' features extracted.')#print extracted feature from image 2

		store_feature = [descriptors2, position2] #store extracted feature from image 2 for next comparision

		if const.DEBUG:
			cv2.imshow('cr1', corner_response1)
			cv2.imshow('cr2', corner_response2)
			cv2.waitKey(0)
		
		#comparing feature of image 1 and image 2 
		print(' - Feature matching .... ', end='', flush=True)
		matchedPairs = featureextractionandmatching.matchingImages(descriptors1, descriptors2, position1, position2, pool, y_range=const.MATCHING_Y_RANGE) #calling matching function to compare extracted feature of both images
		print(str(len(matchedPairs)) +' features matched.') #print how many features matched

		if const.DEBUG:
			plotMatchedPairs(image1, image2, matchedPairs) #plot the matched pairs

		#applying RANSAC on matched pairs and calcuting shifts
		print(' - Best shift using RANSAC ...', end='', flush=True)
		shift = imagestitching.algoRANSAC(matchedPairs, shifts[-1]) #calculating shift of matched pair
		shifts += [shift] #adding shift to shifts for each matched pair
		print('. ', shift) #printing best shift

		#stitching image 1 and image 2  
		print(' - Stitching image .... ', end='', flush=True)
		stitchedImages = imagestitching.stitchingImages(stitchedImages, image2, shift, pool, blending=True)#calling stitching function to stitch image 1 and image 2
		cv2.imwrite(str(i) +'.jpg', stitchedImages) #write stitched images
		print('Image Saved.')


	#performing end to end alignment on all stitched images
	alligned_image = imagestitching.endToEndAlign(stitchedImages, shifts) #calling end to end allignment functions to allign all stitched image
	cv2.imshow('alligned paranomic image',alligned_image)
	cv2.imwrite('alligned_image.jpg', alligned_image) #writing an alligned image
	
	#cropping an alligned image
	cropped_image = imagestitching.cropImage(alligned_image) #calling cropImage functions to cropImage an alligned image
	cv2.imshow('cropped paranomic image',cropped_image)
	cv2.imwrite('cropped_image.jpg', cropped_image)#writing a cropped image
	cv2.waitKey(0)