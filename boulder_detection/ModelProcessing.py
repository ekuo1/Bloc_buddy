import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from tensorflow import keras
#from keras.models import load_model
from pathlib import Path, PureWindowsPath
import os
from openvino.runtime import Core, serialize
import time 


#from RouteDetector import getFirstFrame, viewImage, routeDetector, showPredefinedroute #only used in testing


def getMaskFromModel(first_frame, script_path, compiled_model_ir=None):

    # if first_frame.shape[0] < first_frame.shape[1]: #some of the portrait videos I process are landscape?
    #     first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
    #print(first_frame.shape)
    

    ######################## KERAS #######################
    # Please read google drive instructions 
    # Place the weights in the matching relative path below
    # path = f"{script_path}/boulder_detection/boulder_88_no_batch_norm_16_batch_size_freezeweights_resnet50.h5"
    # model = load_model(path, compile = False)

    # ####################### OPENVINO #################### ##https://docs.openvino.ai/2023.0/home.html
    # Please read google drive instructions
    # Download .bin file from google drive
    # Place the weights in the matching relative path below 
    if compiled_model_ir is None:
        path = f"{script_path}/boulder_detection/model-IR.xml"
        ie = Core()
        model_ir = ie.read_model(model=path)
        model_ir.reshape([-1, 256, 256, 3]) #dynamic first parameter
        compiled_model_ir = ie.compile_model(model=model_ir, device_name="AUTO", config={"PERFORMANCE_HINT":"LATENCY"})

    # Get output layer
    output_layer_ir = compiled_model_ir.output(0)
    # #####################################################

    ########## input first frame of shape (x, y, 3)
    ########## output reconstructed mask of shape (x, y)

    old_width = first_frame.shape[1]
    old_height = first_frame.shape[0] 

    first_frame = cv2.medianBlur(first_frame,5)

    #reshaping frame to be a multiple of 256, to ensure we can extract patches of 256x256
    new_width = (first_frame.shape[1] + 255) // 256 * 256 
    new_height = (first_frame.shape[0] + 255) // 256 * 256


    first_frame = cv2.resize(first_frame, (new_width,new_height))
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # print(first_frame_gray.shape)
    # plt.imshow(first_frame_gray, "gray")
    # plt.show()


    ##### removing glare
    img = first_frame.copy()
    #convert to gray
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #INPAINT
    mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
    result1 = cv2.inpaint(img, mask1, 30, cv2.INPAINT_TELEA) 
    result1_gray = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
    

    #patches = patchify(first_frame_gray, (256, 256), step=256)
    patches = patchify(result1_gray, (256, 256), step=128) #overlapping patches of 128 pixels
    
    #extracting patches from prediction model
    predicted_patches = []
    ##old method
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            
            single_patch = patches[i,j,:,:]
            single_patch_norm = (single_patch.astype('float32')) / 255.
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            single_patch_input = np.stack((single_patch_input,)*3, axis=-1)
            ####single_patch_prediction = (model.predict(single_patch_input, verbose = 0)[0,:,:,0]>0.8).astype(np.uint8) ###keras method
            single_patch_prediction = (compiled_model_ir([single_patch_input])[output_layer_ir])>0.8 ###openvino method
            predicted_patches.append(single_patch_prediction)

    #################################### batch prediction
    # batch_patches = np.stack(patches)
    # batch_patches = batch_patches.reshape(-1, 256, 256)
    # batch_patches = batch_patches[..., np.newaxis]
    # batch_patches = batch_patches.astype('float32') / 255.
    # batch_patches = np.repeat(batch_patches, 3, axis=-1)
    # batch_predictions = model.predict_on_batch(batch_patches)
    # batch_predictions = (batch_predictions[:, :, :, 0] > 0.8).astype(np.uint8)
    # predicted_patches.append(batch_predictions)
    # predicted_patches = np.concatenate(predicted_patches)

    predicted_patches = np.array(predicted_patches)

    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256,256) )
    reconstructed_image = unpatchify(predicted_patches_reshaped, first_frame_gray.shape)
    reconstructed_image = np.where(reconstructed_image == 1, 255, 0).astype(np.uint8) #original reconstructed image
    # end = time.time()
    # print("Inference time is: ", end-start)

    #reconstructing overlapping patches
    overlap_prediction = reconstructingPrediction(predicted_patches_reshaped, reconstructed_image)
    #reconstructed_image = cv2.resize(reconstructed_image, (old_width,old_height))
    overlap_prediction = cv2.resize(overlap_prediction, (old_width,old_height))
    print("overlap_prediction", overlap_prediction.shape)
    

    return overlap_prediction


### reconstruciting the overlapping patches by using bitwise_and operation on the overlapping sections. 
def reconstructingPrediction(predicted_patches_reshaped, reconstructed_image):
    predicted_patches_horizontal = []
    predicted_patches_reshaped = np.where(predicted_patches_reshaped == 1, 255, 0).astype(np.uint8)
    #viewImage(reconstructed_image)

    for j in range(0, predicted_patches_reshaped.shape[0]-1):
        for i in range(0, predicted_patches_reshaped.shape[1]-1):
            patchleft = predicted_patches_reshaped[j][i]
            patchright = predicted_patches_reshaped[j][i+1]
            #viewImage(patchleft)
            patchhorizontal = np.bitwise_and(patchleft[0:128, 128:256], patchright[0:128, 0:128]) ##bitwise comparison of overlapping patches
            
            predicted_patches_horizontal.append(patchhorizontal)

    newh = predicted_patches_reshaped.shape[0]-1
    neww = predicted_patches_reshaped.shape[1]-1
    #print(newh, neww)
    predicted_patches_horizontal = np.array(predicted_patches_horizontal)
    predicted_patches_horizontal_reshaped = np.reshape(predicted_patches_horizontal, (newh,neww, 128,128)) #new patches are stitched back together
    reconstructed_image_bitmap = unpatchify(predicted_patches_horizontal_reshaped, (newh*128, neww*128))
    #viewImage(reconstructed_image_bitmap) 
    
    copy = reconstructed_image.copy()
    
    copy[0:(newh*128), 128:(neww*128 + 128)] = reconstructed_image_bitmap
    
    return copy   


#using watershed model and erosion and dialation to remove boulders that are touching
def maskPostProcessing(mask): 

    ##filling  contours
    contours, hierarchy =  cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask_copy = mask.copy()
    mask_copy = cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_copy, contours, -1, (0,255,0),2, maxLevel=2)
    filled_mask = np.zeros_like(mask)
    cv2.fillPoly(filled_mask, contours, 255)


    ##seperating boulders that have joined together using opencv watershed method
    ret, thresh = cv2.threshold(filled_mask,0,255,cv2.THRESH_OTSU)
    
    #noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    #sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=1)

    #sure foreground area
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(opening,kernel,iterations = 3)
    erosion = np.uint8(erosion)
    
    #finding unknown region
    unknown = cv2.subtract(sure_bg,erosion)
    ret, markers = cv2.connectedComponents(erosion)
    markers = markers+1
    markers[unknown==255] = 0

    filled_mask = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(filled_mask,markers)
    filled_mask[markers == -1] = [0,0,0]


    kernel = np.ones((3,3),np.uint8)
    filled_mask = cv2.erode(filled_mask,kernel,iterations = 1)
    
    
    return np.uint8(filled_mask)


def findBottomEdge(image, contours_dict):

    # #initialise variables
    closest_point = None
    closest_distance = float('inf')

    # #iterate through each contour
    for _, contours in contours_dict.items():
        for contour in contours:
            for point in contour:
                x, y = point[0]
                distance_to_bottom = image.shape[0]-y
                if distance_to_bottom < closest_distance:
                    closest_distance = distance_to_bottom
                    closest_point = (x, distance_to_bottom)

    

    horizontal_line = int(np.floor(closest_point[1] + 0.05*(image.shape[0])))
    #print("Estimate of floor to wall:", horizontal_line)

    # point1 = (0, horizontal_line)
    # point2 = (image.shape[1], horizontal_line)
    # cv2.line(image, point1, point2, (0, 255, 0), 2)
    return horizontal_line


# video_path = 'C:/Users/Sarah/Documents/Uni/BoulderBuddy/FYP/test_examples/yellow5.mp4'
# # # # output_path = './BoulderWall/'

# # # #model = load_model('C:/Users/Sarah/Documents/Uni/BoulderBuddy/FYP/boulder_detection/boulder_88_no_batch_norm_16_batch_size_freezeweights_resnet50.h5', compile = False)
# script_path = Path(__file__).parent

# first_frame = getFirstFrame(video_path)
# # if first_frame.shape[0] < first_frame.shape[1]: #some of the portrait videos I process are landscape?
# #         first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
# #first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
# viewImage(first_frame)

# start = time.time()
# model_mask = getMaskFromModel(first_frame, script_path)
# end = time.time()
# print(end-start)
# viewImage(model_mask)
# model_mask_postpro = maskPostProcessing(model_mask)
# print("post processing")
# viewImage(model_mask_postpro)

# # print(first_frame.shape)

# rout_dic = routeDetector(first_frame, model_mask_postpro)

# findBottomEdge(first_frame, rout_dic)

# # #goes through all colours to find routes
# showPredefinedroute(first_frame, rout_dic)


