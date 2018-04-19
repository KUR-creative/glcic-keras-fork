from utils import random_masked
from keras.models import load_model
import numpy as np
import cv2
# create interactive UI.
# get mask from UI.
# run complnet.
# print result of complnet.

imgpath = './mse_model.png'

origin = cv2.imread(imgpath)
h,w = origin.shape[:2]
mean_pixel_value = np.mean(origin)

mask, mask_yx, mask_hw = random_masked((h,w,3), 100, 200)  
(mY,mX), (mH,mW) = mask_yx, mask_hw

complnet_input = np.copy(origin)
complnet_input[mY:mY+mH,mX:mX+mW] = mean_pixel_value

not_mask = np.logical_not(mask)#.reshape((h,w,1))#.astype(np.float32)
holed_origin = not_mask * origin

model = load_model('./output/complnet_199.h5')
completed = model.predict([holed_origin.reshape((1,h,w,3)), 
                           complnet_input.reshape((1,h,w,3)), 
                           mask.reshape((1,h,w,3))])

#print(mask.dtype)
cv2.imshow('origin',origin); cv2.waitKey(0)
cv2.imshow('holed_origin',holed_origin); cv2.waitKey(0)
cv2.imshow('mask',mask.astype(np.float32)); cv2.waitKey(0)
cv2.imshow('completed',completed); cv2.waitKey(0)
