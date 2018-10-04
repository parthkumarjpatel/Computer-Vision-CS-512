#Name : Parthkumar Patel
#CWID : A20416508
#SEMESTER : FALL 2018
#COURSE : CS-512 : COMPUTER VISION
#ASSIGNMENT : 3 - Programming Questions

import cv2
import sys
import numpy as np
import random
from scipy import ndimage
from matplotlib import pyplot as plt

image = cv2.imread("lenna.jpg",1) # reading an image as a 3 channel color image
original_image = image
cv2.imshow('image',image) # showing an image
        
print("This is your original image shown in another window. \nPress h for short description of commands.")

 
while True:
    user_input = chr(cv2.waitKey()) # taking an input command from user 
    
    if user_input== 'i':  #reload the original image
        print("You have selected command : i")
        image=cv2.imread('lenna.jpg',1)
        
    if user_input=='w': # save the curent image
        print("You have selected command : w")
        cv2.imwrite('out.jpg',image)
        
    if user_input == 'g' : # convert the image to grayscale
        image = original_image 
        print("You have selected command : g")
        image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        
        
    if user_input=='G': # convert the image to grayscale using own implementation
        image = original_image   
        print("You have selected command : G")
        def grayConversion(image):
            grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0] # converting the image and multiplying each channel
            gray_img = grayValue.astype(np.uint8)
            return gray_img
        image = grayConversion(image) # converted gray image
        
    if user_input == 'c':
        image = original_image
        print("You have selected command : c")
		
		#making blue and green parts zero in red channel
        r = image.copy() 
        r[:,:,0] = 0
        r[:,:,1] = 0
        
		#making red and green parts zero in blue channel
        b = image.copy() 
        b[:,:,1] = 0
        b[:,:,2] = 0
        
		#making red and blue parts zero in green channel
        g = image.copy()
        g[:,:,0] = 0
        g[:,:,2] = 0
        

        image = random.choice([b,g,r]) # taking random channel every time
       
    if user_input == 's': # converting into grayscale and smoothing the image
        image = original_image
        print("You have selected command : s")
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY ) #converting into grayscale
        image = cv2.blur(image_gray,(3,3)) #smoothing image
                       
    if user_input == 'S':
        image = original_image
        print("You have selected command : S")
		
		#function to smooth
        def convolve2d(image_smooth, kernel):
            kernel = np.flipud(np.fliplr(kernel))    
            output = np.zeros_like(image_smooth)    
			
            # Add zero padding to the input image
            image_padded = np.zeros((image_smooth.shape[0] + 2, image_smooth.shape[1] + 2))   
            image_padded[1:-1, 1:-1] = image_smooth

            for x in range(image_smooth.shape[1]): 
                # Loop over every pixel of the image
                for y in range(image_smooth.shape[0]):
                    output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
            return output
        
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY )
        kernel = (np.array([[1,1,1],[1,1,1],[1,1,1]])/9) # 3*3 kernal 
        image= convolve2d(image_gray,kernel)
        
        
        
    if user_input == 'd': #downsampling the image without smoothing
        image = original_image
        print("You have selected command : d")
        image = cv2.pyrDown(image,2) # downsampling by factor of 2
        
        
        
    if user_input == 'D': #downsampling the image with smoothing
        image = original_image
        print("You have selected command : D")
        image_withsmooth = cv2.blur(image,(3,3))
        image =cv2.pyrDown(image_withsmooth,2) #downsampling the image with smoothing by factor of 2
        
        
        
    if user_input == 'x': # performing convolution on image
        image = original_image
        print("You have selected command : x")
        image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        der=[[-1,0,1],[-2,0,2],[-1,0,1]] #  x derivative filter
        xder=np.array(der)
        normimage=np.zeros((1920,1080))
        newimagex=ndimage.convolve(image,xder,mode='constant') # convolution
        normaimagex=cv2.normalize(newimagex,normimage,0,255,cv2.NORM_MINMAX) # normalizing the image
        image = normaimagex
        
        
    if user_input== 'y': # performing convolution on image
        image = original_image
        print("You have selected command : y")
        image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        der=[[-1,-2,21],[0,0,0],[1,2,1]] #  y derivative filter
        xder=np.array(der)
        normimage=np.zeros((1920,1080))
        newimagey=ndimage.convolve(image,xder,mode='constant') #convolution
        normaimagey=cv2.normalize(newimagey,normimage,0,255,cv2.NORM_MINMAX) #normalizing the image
        image = normaimagey
        
        
    if user_input == 'm': # magnitude of the gradient nomalized
        image = original_image
        print("You have selected command : m")
        def gramag(x,y,image):
                sobel1=cv2.Sobel(image,cv2.CV_64F,x,y,ksize=5) # sobel filter
                abs_val=np.absolute(sobel1)
                s8U=np.uint8(abs_val)
                return s8U
        image = gramag(1,1,image) 
        
    if user_input == 'p': #plotting the gradient vectors of image
        image = original_image 
        print("You have selected command : p")
        image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # original gray scale image
        laplacian = cv2.Laplacian(image,cv2.CV_64F) # laplacian image
        sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5) # sobel x image
        sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5) # sobel y image
        plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        plt.show()
        
    if user_input =='r': # rotaing an image using an angle theta
        image = original_image
        print("You have selected command : r")
        theta= float(input('Enter an angle --->' )) # asking for theta value from user
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY )
        num_rows, num_cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), theta, 1) # using function to rotate matrix
        image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
        
    if user_input == 'h':
        print("You have selected command : h")
        print('\n Press i to reload the original image \n Press w to save the curent image \n Press g to view the grayscale image using opencv function \n Press G to view the custom grayscale image \n Press c to view the image in various color channels\n Press s to smooth the image with trackbar functionality \n Press S to view custom smoothing of image \n Press d to downsample the image by a factor of 2 without smoothing \n Press D to downsample the image by a factor of 2 with smoothing \n Press x to perform convolution with x derivative filter with normalization\n Press y to perform convolution with y derivative filter with normalization \n Press m to show the magnitude of the gradient nomalized \n Press p to convert image to grayscale and plot the gradient vectors \n Press r to rotate the image with an angle theta \n')
        
        
    cv2.imshow('image',image) # showing an image after performing above mentioned operations
