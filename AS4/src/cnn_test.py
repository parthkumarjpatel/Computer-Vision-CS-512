#Name : Parthkumar Patel
#CWID : A20416508
#SEMESTER : FALL 2018
#COURSE : CS-512 : COMPUTER VISION
#ASSIGNMENT : 4 - Programming Questions

import cv2
import numpy as np
from keras.models import load_model
import sys
while True:
    print("enter name of the image with .jpg extension or enter 'q' for exit: ")
    input_image = input()


    if input_image== 'q':
        sys.exit()
    else:
        #read the image from input given by user
        img = cv2.imread(input_image,1)
        #show the image
        cv2.imshow('original image',img)

        #resize the image to our mode's size
        img = cv2.resize(img,(28,28))

        #convert the image into grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #using GaussianBlur() and adaptiveThreshold()
        img= cv2.GaussianBlur(img,(3,3),0)
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_BINARY_INV,11,2)
        #show the binary image
        cv2.imshow('binary image',img)
        x = img.reshape(1,28,28,1)

        #load the CNN model and use predict function to predict the value
        model =load_model('local1.h5')
        y = model.predict(x)
        print(y)
        if y<=0.5 :
            print("even")
        else :
            print("odd")