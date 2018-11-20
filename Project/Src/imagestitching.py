#Name : Parthkumar Patel, Arpit Hasmukhbhai Patel
#CWID : A20416508, A20424085
#SEMESTER : FALL 2018
#COURSE : CS-512 : COMPUTER VISION
#Project : Panoramic Image Stitching

import numpy as np
import cv2
import constant as const

"""
Find best shift of matched pairs using RANSAC

Args:
    matchedPairs: matched pairs of feature's positions
    previousShift: previous shift, for checking shift direction.

Returns:
    Best shift [y x]. ex. [4 234]

Raise:
    ValueError: Shift direction NOT same as previous shift.
"""
def algoRANSAC(matchedPairs, previousShift):
    matchedPairs = np.asarray(matchedPairs)
    
    selectRandom = True if len(matchedPairs) > const.RANSAC_K else False

    bestShift = [] #model parameters which best fit the data
    K = const.RANSAC_K if selectRandom else len(matchedPairs) #maximum number of iterations allowed in the algorithm
    thresholdDistance = const.RANSAC_THRES_DISTANCE
    
    max_inliner = 0
    for k in range(K):
        # Random pick a pair of matched feature
        idx = int(np.random.random_sample()*len(matchedPairs)) if selectRandom else k #n randomly selected values from data
        sample = matchedPairs[idx]
        
        # fit the warp model
        shift = sample[1] - sample[0]
        
        # calculate inliner points
        shifted = matchedPairs[:,1] - shift
        difference = matchedPairs[:,0] - shifted
        
        inliner = 0
        for diff in difference: #for every point in data not in maybeInliers 
            if np.sqrt((diff**2).sum()) < thresholdDistance: #if point fits maybeModel with an error smaller than thresholddistance
                inliner = inliner + 1 #add point to alsoInliers
        
        if inliner > max_inliner: #if the number of elements in alsoInliers is > d
            max_inliner = inliner 
            bestShift = shift #a measure of how well betterModel fits these points

    if previousShift[1]*bestShift[1] < 0:
        print('\n\nBest shift:', bestShift)
        raise ValueError('Shift direction NOT same as previous shift.')

    return bestShift


"""
Stitch two image with blending.

Args:
    image1: first image
    image2: second image
    shift: the relative position between image1 and image2
    pool: for multiprocessing
    blending: using blending or not

Returns:
    A stitched image
"""
def stitchingImages(image1, image2, shift, pool, blending=True):
    padding = [
        (shift[0], 0) if shift[0] > 0 else (0, -shift[0]),
        (shift[1], 0) if shift[1] > 0 else (0, -shift[1]),
        (0, 0)
    ]
    shiftedImage1 = np.lib.pad(image1, padding, 'constant', constant_values=0) #add padding to image1

    # cut out unnecessary region
    split = image2.shape[1]+abs(shift[1])
    splited = shiftedImage1[:, split:] if shift[1] > 0 else shiftedImage1[:, :-split]
    shiftedImage1 = shiftedImage1[:, :split] if shift[1] > 0 else shiftedImage1[:, -split:]

    h1, w1, _ = shiftedImage1.shape
    h2, w2, _ = image2.shape
    
    inverseShift = [h1-h2, w1-w2]
    inversePadding = [
        (inverseShift[0], 0) if shift[0] < 0 else (0, inverseShift[0]),
        (inverseShift[1], 0) if shift[1] < 0 else (0, inverseShift[1]),
        (0, 0)
    ]
    shiftedImage2 = np.lib.pad(image2, inversePadding, 'constant', constant_values=0) #add padding to image2

    direction = 'left' if shift[1] > 0 else 'right' #direction is set left if shift is greater than 0

    if blending:
        seam_x = shiftedImage1.shape[1]//2
        tasks = [(shiftedImage1[y], shiftedImage2[y], seam_x, const.ALPHA_BLEND_WINDOW, direction) for y in range(h1)]
        shiftedImage1 = pool.starmap(alphaBlend, tasks)
        shiftedImage1 = np.asarray(shiftedImage1)
        shiftedImage1 = np.concatenate((shiftedImage1, splited) if shift[1] > 0 else (splited, shiftedImage1), axis=1)
    else:
        raise ValueError('blending=False')

    return shiftedImage1 #return shifted Image 1

def alphaBlend(row1, row2, seam_x, window, direction='left'):
    #if direction is right then row exchanged
    if direction == 'right':
        row1, row2 = row2, row1

    newRow = np.zeros(shape=row1.shape, dtype=np.uint8)

    for x in range(len(row1)):
        color1 = row1[x]
        color2 = row2[x]
        if x < seam_x-window:
            newRow[x] = color2
        elif x > seam_x+window:
            newRow[x] = color1
        else:
            ratio = (x-seam_x+window)/(window*2)
            newRow[x] = (1-ratio)*color2 + ratio*color1

    return newRow

"""
End to end alignment of an panormaic image

Args:
    image: panoramaic image
    shifts: all shifts for each image in panorama

Returns:
    A image that fixed the y-asix shift error
"""
def endToEndAlign(image, shifts):
    sumOfY, sumOfX = np.sum(shifts, axis=0)

    yShift = np.abs(sumOfY)
    columnShift = None

    # same sign
    if sumOfX*sumOfY > 0:
        columnShift = np.linspace(yShift, 0, num=image.shape[1], dtype=np.uint16)
    else:
        columnShift = np.linspace(0, yShift, num=image.shape[1], dtype=np.uint16)

    alignedImage = image.copy()
    #allign all the images
    for x in range(image.shape[1]):
        alignedImage[:,x] = np.roll(image[:,x], columnShift[x], axis=0)

    return alignedImage #return an aligned image


"""
crop the black border in image

Args:
    image: a panoramaic image

Returns:
    Cropped image
"""
def cropImage(image):
    _, thresholdforCropping = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    upperLimit, lowerLimit = [-1, -1]

    blackPixelNumberThreshold = image.shape[1]//100

    #cropping a black border from upper part of an image
    for y in range(thresholdforCropping.shape[0]):
        if len(np.where(thresholdforCropping[y] == 0)[0]) < blackPixelNumberThreshold:
            upperLimit = y
            break
        
    #cropping a black border from lower part of an image
    for y in range(thresholdforCropping.shape[0]-1, 0, -1):
        if len(np.where(thresholdforCropping[y] == 0)[0]) < blackPixelNumberThreshold:
            lowerLimit = y
            break

    return image[upperLimit:lowerLimit, :] #return a cropped image
