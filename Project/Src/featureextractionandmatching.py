#Name : Parthkumar Patel, Arpit Hasmukhbhai Patel
#CWID : A20416508, A20424085
#SEMESTER : FALL 2018
#COURSE : CS-512 : COMPUTER VISION
#Project : Panoramic Image Stitching

import cv2
import numpy as np
import constant as const

#computing raw response of Structure tensor setup M
def computeRawResponse(xx_row, yy_row, xy_row, k):
    row_response = np.zeros(shape=xx_row.shape, dtype=np.float32)

    #raw response = det(M) = k * trace(M)^2 
    for x in range(len(xx_row)):
        det_M = xx_row[x]*yy_row[x] - xy_row[x]**2 
        trace_M = xx_row[x] + yy_row[x] #trace_M = lambda 1 + lambda 2 
        R = det_M - k*trace_M**2
        row_response[x] = R

    return row_response

"""
Harris corner detection

Args:
    image: input image
    pool: for multiprocessing
    k: harris corner constant value
    block_size: harris corner windows size

Returns:
    A corner response matrix with the same  width and height as input image
"""
def harrisCornerDetection(image, pool, k=0.04, block_size=2):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting image to grayscale 
    grayImage = np.float32(grayImage)/255 

    cornerResponse = np.zeros(shape=grayImage.shape, dtype=np.float32)
    
    height, width, _ = image.shape #getting height and width of an image

    #Spatial derivative calculation
    dx = cv2.Sobel(grayImage, -1, 1, 0)
    dy = cv2.Sobel(grayImage, -1, 0, 1)
    I_xx = dx*dx
    I_yy = dy*dy
    I_xy = dx*dy
    
    #Blurs an image using the box filter
    covariance_xx = cv2.boxFilter(I_xx, -1, (block_size, block_size), normalize=False) 
    covariance_yy = cv2.boxFilter(I_yy, -1, (block_size, block_size), normalize=False)
    covariance_xy = cv2.boxFilter(I_xy, -1, (block_size, block_size), normalize=False)

    #Harris corner response calculation
    cornerResponse = pool.starmap(computeRawResponse, [(covariance_xx[y], covariance_yy[y], covariance_xy[y], k) for y in range(height)])
            
    return np.asarray(cornerResponse)
    
"""
Extract descritptor from corner response image

Args:
    image : input image
    cornerResponse: corner response matrix
    threshlod: only corner response > 'max_corner_response*threshold' will be extracted
    kernel: descriptor's window size, the descriptor will be kernel^2 dimension vector 

Returns:
    A pair of (descriptors, positions)
"""
def extractDescofCornerResponse(image, cornerResponse, threshold=0.01, kernel=3):
    height, width = cornerResponse.shape

    # Reducing corner
    features = np.zeros(shape=(height, width), dtype=np.uint8)
    features[cornerResponse > threshold*cornerResponse.max()] = 255

    # Trimming feature on edges of image
    features[:const.FEATURE_CUT_Y_EDGE, :] = 0  
    features[-const.FEATURE_CUT_Y_EDGE:, :] = 0
    features[:, -const.FEATURE_CUT_X_EDGE:] = 0
    features[:, :const.FEATURE_CUT_X_EDGE] = 0
    
    # Reducing features using local maximum
    window=3
    for y in range(0, height-10, window):
        for x in range(0, width-10, window):
            if features[y:y+window, x:x+window].sum() == 0:
                continue
            block = cornerResponse[y:y+window, x:x+window]
            maximum_y, maximum_x = np.unravel_index(np.argmax(block), (window, window))
            features[y:y+window, x:x+window] = 0
            features[y+maximum_y][x+maximum_x] = 255

    featurePositions = []
    featureDescriptions = np.zeros(shape=(1, kernel**2), dtype=np.float32)
    
    half_k = kernel//2
    for y in range(half_k, height-half_k):
        for x in range(half_k, width-half_k):
            if features[y][x] == 255:
                featurePositions += [[y, x]]
                desc = cornerResponse[y-half_k:y+half_k+1, x-half_k:x+half_k+1]
                featureDescriptions = np.append(featureDescriptions, [desc.flatten()], axis=0)
                
    return featureDescriptions[1:], featurePositions

"""
Matching two groups of descriptors

Args:
    descriptor1: descriptor of image 1 
    descriptor2: descriptor of image 2
    featurePosition1: descriptor1's corrsponsed position
    featurePosition2: descriptor2's corrsponsed position
    pool: for mulitiprocessing
    y_range: restrict only to match y2-y_range < y < y2+y_range

Returns:
    matched position pairs, it is a Nx2x2 matrix
"""
def matchingImages(descriptor1, descriptor2, featurePosition1, featurePosition2, pool, y_range=10):
    TASKS_NUM = 32 

    partitionDescriptors = np.array_split(descriptor1, TASKS_NUM) #dividing descriptor for multiprocessing
    partitionPositions = np.array_split(featurePosition1, TASKS_NUM) #dividing feature position for multiprocessing

    subTasks = [(partitionDescriptors[i], descriptor2, partitionPositions[i], featurePosition2, y_range) for i in range(TASKS_NUM)]
    results = pool.starmap(computeMatchingOfTwoImages, subTasks) #compute matching of two images
    
    matchedPairs = [] #intializing matched pairs
    for res in results:
        if len(res) > 0:
            matchedPairs += res

    return matchedPairs #returning matched pairs
"""
compute Matching of two images

Args:
    descriptor1: descriptor of image 1 
    descriptor2: descriptor of image 2
    featurePosition1: descriptor1's corrsponsed position
    featurePosition2: descriptor2's corrsponsed position
    y_range: restrict only to match y2-y_range < y < y2+y_range

Returns:
    refined matched position pairs, it is a Nx2x2 matrix
"""
def computeMatchingOfTwoImages(descriptor1, descriptor2, featurePosition1, featurePosition2, y_range=10):
    matchedPairs = [] #intializing matched pairs array
    matchedPairsRank = [] #intializing matched pairs rank array
    
    for i in range(len(descriptor1)):
        distances = []
        y = featurePosition1[i][0] #descriptor1's corrsponsed position
        for j in range(len(descriptor2)):
            diff = float('Inf')
            
            # only compare features that have similiar y-axis 
            if y-y_range <= featurePosition2[j][0] <= y+y_range:
                diff = descriptor1[i] - descriptor2[j]
                diff = (diff**2).sum()
            distances += [diff]

        sortedIndex = np.argpartition(distances, 1)
        localOptimal = distances[sortedIndex[0]]
        localOptimal2 = distances[sortedIndex[1]]
        #exchanging localoptimal and localoptimal2
        if localOptimal > localOptimal2:
            localOptimal, localOptimal2 = localOptimal2, localOptimal
        
        if localOptimal/localOptimal2 <= 0.5:
            pairedIndex = np.where(distances==localOptimal)[0][0]
            pair = [featurePosition1[i], featurePosition2[pairedIndex]]
            matchedPairs += [pair]
            matchedPairsRank += [localOptimal]

    # Refining sorted pairs
    sortedrankIndex = np.argsort(matchedPairsRank)
    sortedMatchPairs = np.asarray(matchedPairs)
    sortedMatchPairs = sortedMatchPairs[sortedrankIndex]


    refinedMatchedPairs = [] #intializing refined matched pairs
    for item in sortedMatchPairs:
        duplicated = False
        for refined_item in refinedMatchedPairs:
            if refined_item[1] == list(item[1]):
                duplicated = True
                break
        if not duplicated:
            refinedMatchedPairs += [item.tolist()]
            
    return refinedMatchedPairs #returning refined matched pairs
