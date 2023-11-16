from matplotlib import pyplot as plt
import cv2
#from IPython import display
import numpy as np
from scipy.signal import find_peaks

#gets the first frame
def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("first_frame.jpg", image)  # save frame as JPEG file
        return image


#homography for stabilisation
def homographyMatrix(img1, original_img2):

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(original_img2, cv2.COLOR_BGR2GRAY)

    #sift
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # Match keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)

    # Sort matches
    matches = sorted(matches, key = lambda x:x.distance)

    # Find keypoint correspondences
    X1 = np.vstack([kp1[match.queryIdx].pt for match in matches])
    X2 = np.vstack([kp2[match.trainIdx].pt for match in matches])

    # Estimate homograpahy using opencv - 
    Hcv, mask = cv2.findHomography(X2, X1, cv2.RANSAC, 5.0, maxIters = 4000)

    # is_good_or_bad = is_good_homography(Hcv, 0.2)
    # if is_good_or_bad == False:
    #     print(":( bad homography transformation")

    im_warp = cv2.warpPerspective(original_img2,Hcv,(img1.shape[1],img1.shape[0]))

    return im_warp, Hcv

def is_good_homography(H, width, height, threshold=0.1):
    # checking that they are unit vectors
    if abs(H[2, 2] - 1) > threshold:
        return False
    
    # checking that they are unit vectors
    # if np.linalg.det(H) < 0.8:
    #     print("b")
    #     return False
    
    if test_orietation_homography(H, width, height) == False:
        return False
    
    if test_scale_homography(H, width, height) == False:
        return False
    
    return True

def test_orietation_homography(homography, width, height):
    # Define the original quadrilateral points
    points = [(0, 0), (width, 0), (width, height), (0, height)]

    # convert points to coordinates
    points_homog = np.array(points + [points[0]], dtype=np.float32)
    points_homog = np.expand_dims(points_homog, axis=1)

    # homography transformation
    transformed_points_homog = cv2.perspectiveTransform(points_homog, homography)

    # cartesian coordinates
    transformed_points = transformed_points_homog.squeeze(axis=1)

    # Calculate signed area of the transformed polygon
    area = 0.5 * np.sum(
        transformed_points[:-1, 0] * transformed_points[1:, 1] -
        transformed_points[:-1, 1] * transformed_points[1:, 0]
    )

    if area > 0:
        return True
    elif area < 0:
        return False
    else:
        return False

def test_scale_homography(homography, width, height, expected_scale_factor=1):
    #define the original quadrilateral points
    points = [(0, 0), (width, 0), (width-1, height), (0, height)]

    #convert points to coordinates
    points_homog = np.array(points + [points[0]], dtype=np.float32)
    points_homog = np.expand_dims(points_homog, axis=1)

    #homography transformation
    transformed_points_homog = cv2.perspectiveTransform(points_homog, homography)

    #convert back to cartesian coordinates
    transformed_points = transformed_points_homog.squeeze(axis=1)

    #calculate areas of the original and transformed quadrilaterals
    original_area = cv2.contourArea(points_homog.squeeze())
    transformed_area = cv2.contourArea(transformed_points)

    #calculate the ratio of the areas
    area_ratio = transformed_area / original_area

    if 0.9 < area_ratio < 1.1:
        return True
    else:
        return False

def homography_different(homography_now, homography_prev, threshold = 2):
    
    diff_matrix = homography_now - homography_prev
    norm_diff = np.linalg.norm(diff_matrix, 'fro') #forbenius norm
    return norm_diff > threshold


#generalised colour upper and lower bounds 
color_dict_HSV = {
              'red': [[14, 255, 220], [0, 30, 70]],
              'red2': [[180, 255, 220], [175, 30, 70]],
              'green': [[95, 255, 220], [36, 30, 60]],
              'blue': [[122, 255, 220], [95, 30, 70]],
              'yellow': [[35, 255, 220], [15, 30, 25]],
              'purple': [[155, 255, 220], [122, 30, 70]],
              'pink': [[174, 255, 220], [155, 30, 70]],
              'black': [[180, 255, 140], [0, 0, 0]]}

            #   'red': [[14, 255, 255], [0, 80, 70]],
            #   'red2': [[180, 255, 255], [175, 40, 70]],
            #   'green': [[95, 255, 255], [36, 40, 60]],
            #   'blue': [[120, 255, 255], [95, 60, 70]],
            #   'yellow': [[35, 255, 255], [15, 70, 50]],
            #   'purple': [[150, 255, 255], [120, 40, 50]],
            #   'pink': [[174, 255, 255], [150, 40, 80]], 
            #   'black': [[180, 255, 20], [0, 0, 0]],
            #   'white': [[180, 18, 255], [0, 0, 231]]}




def viewImage(image):

    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Display', 600, 500)
    cv2.imshow('Display', image)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#colour masking the image
def colour_mask(colour_dictionary, colour, image):

    low = np.array(colour_dictionary[colour][1] )
    high = np.array(colour_dictionary[colour][0])
    curr_mask = cv2.inRange(image, low, high)

    return curr_mask

#outputting binary mask with the location of boulders
def mask_segmentation(image, color):

    image = cv2.medianBlur(image,5)
    viewImage(image)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == 'red':
        final_mask = colour_mask(color_dict_HSV, color, hsv_img) | colour_mask(color_dict_HSV, 'red2', hsv_img)
    else:
        final_mask = colour_mask(color_dict_HSV, color, hsv_img)

    #viewImage(curr_mask3)
    hsv_img[final_mask > 0] = ([255,255,255])
    hsv_img[final_mask == 0] = ([0,0,0])

    viewImage(hsv_img) 
    
    return hsv_img


#filtering by hue in hsv
def get_key_for_value_in_range(dictionary, value, valvalue):
    for key, val in dictionary.items():
        if ((val[0][0] >= value >= val[1][0]) & (valvalue > 50)): ##Value of < 50 is usually black 
            return key
    return None

#filtering by value in hsv
def black_hsv(dictionary, value, huevalue):
    val = dictionary['black']
    if ((val[0][2] >= value >= val[1][2])):
            return 'black'
    return None


# recieve a mask and an image 
# overlay the mask with the image 
#
# output: route_dict = {key = colour, value = list of contours}
#
def routeDetector(rgb_image, mask):

    route_dict = {
              'red': [],
              'red2': [],
              'green': [],
              'blue': [],
              'yellow': [],
              'purple': [],
              'pink': [],
              'black': [], 
              }

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    #viewImage(gray_image)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #viewImage(gray_mask)
    ret, threshold = cv2.threshold(gray_mask, 90, 255, 0)
    
    contours, hierarchy =  cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (0,255,0),2, maxLevel=2)
    #viewImage(mask)

    overlayed_mask = cv2.bitwise_or(rgb_image,rgb_image,mask = threshold)
    #viewImage(overlayed_mask)
    #print(overlayed_mask.shape)


    hsv_image = cv2.cvtColor(overlayed_mask, cv2.COLOR_BGR2HSV)

    total_area = hsv_image.shape[0]*hsv_image.shape[1]
    print(total_area)
    threshold_area = 0.00001*total_area     
    threshold_area_max = 0.1*total_area
    filtered_contours = []   

    for c in contours: 
        area = cv2.contourArea(c)         
        if (area > threshold_area) & (area < threshold_area_max) :                   
            filtered_contours.append(c)
    

    for fc in filtered_contours: 
        mask = np.zeros(gray_image.shape, np.uint8)
        cv2.drawContours(mask, [fc], 0, 255, -1)
        #viewImage(mask) 

        mask2 = np.zeros(gray_image.shape, np.uint8)
        mask2.fill(255)

        new_mask = cv2.bitwise_and(mask,mask, mask = mask2)
        
        new_image = cv2.bitwise_and(rgb_image, rgb_image, mask = new_mask)
        #viewImage(new_image)

        hist_h = cv2.calcHist([hsv_image],[0],mask,[180],[1,180])
        max_hue_vale = (np.argmax(hist_h))
        #print(max_hue_vale)

        hist_v = cv2.calcHist([hsv_image],[2],mask,[180],[1,180])
        max_val_vale = (np.argmax(hist_v))
        #print(max_hue_vale)

        # if get_key_for_value_in_range(color_dict_HSV, max_hue_vale) == "blue":
        #     viewImage(new_image)
        #     print(max_val_vale)
        #     #clusteredBoudlers(max_hue_vale, hist_h)
        #     plt.plot(hist_v)
        #     plt.show()

        key = get_key_for_value_in_range(color_dict_HSV, max_hue_vale, max_val_vale)
        keyb = black_hsv(color_dict_HSV, max_val_vale, max_hue_vale)
        clusteredb = clusteredBoudlers(max_hue_vale, hist_h, max_val_vale)
        #print(key)

        if clusteredb != None:
            for cluster_name in clusteredb: 
                route_dict[cluster_name].append(fc)
        
        if key != None:
            route_dict[key].append(fc)

        if keyb != None:
            route_dict[keyb].append(fc)

    #100% of red and pink
    combined_arrays = route_dict['red'] + route_dict['red2'] + route_dict['pink']
    route_dict['red'] = combined_arrays
    route_dict['red2'] = combined_arrays
    route_dict['pink'] = combined_arrays


    return route_dict

def clusteredBoudlers(max, hist, valvalue): 
    peaks, _ = find_peaks(hist.flatten(), height=100, distance=20)
    original = get_key_for_value_in_range(color_dict_HSV, max, valvalue)
    clustered = []
    if len(peaks) > 1:
        for m in peaks:
            if original != get_key_for_value_in_range(color_dict_HSV, m, valvalue):
                clustered.append(get_key_for_value_in_range(color_dict_HSV, m, valvalue))
    return clustered



#for debugging purposes
#displays all found routes
def showPredefinedroute(image, dict):

    contour_color = {
              'black': (0, 0, 0), 
              'red': (0,0,255),
              'red2': (0,0,255),
              'green': (0, 255, 0),
              'blue': (255, 0, 0),
              'yellow': (203, 255, 255),
              'purple': (128, 0, 128),
              'pink': (203, 192, 255),
              }
    
    for key, val in dict.items():
        #image = cv2.imread(path)
        color = contour_color[key]
        for contour in val:
            cv2.drawContours(image, [contour],-1, color,2, maxLevel=2)
        
        cv2.namedWindow(str(key), cv2.WINDOW_NORMAL)
        cv2.resizeWindow(str(key), 600, 500)
        cv2.imshow(str(key), image)  
        cv2.waitKey(0)
        cv2.destroyAllWindows()


##################################################################################

# path = 'C:/Users/Sarah/Documents/Uni/FYP/TestNeuralNetImages/yellow_jugs.jpg'
# image = cv2.imread(path)
# viewImage(image)
# imagemask = cv2.imread('C:/Users/Sarah/Documents/Uni/FYP/TestNeuralNetImages/yellowjugs2.jpg')
# imagemask = cv2.resize(imagemask, (1080, 1920))
# #imagemask = np.where(imagemask == 1, 255, 0).astype(np.uint8)
# viewImage(imagemask)
# print(image.shape, imagemask.shape)
# #imagemask = cv2.cvtColor(imagemask, cv2.COLOR_BGR2GRAY)
# # #create own mask of the colour of choice
# #imagemask = mask_segmentation(image, 'blue')

# # #segments the route
# rout_dic = routeDetector(image, imagemask)

# # #goes through all colours to find routes
# showPredefinedroute(image, rout_dic)
