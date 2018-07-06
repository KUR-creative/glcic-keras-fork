import numpy as np
import cv2
from networks import completion_net, discrimination_net
from keras.layers import Input, Add, Multiply, merge
from keras.models import Model
from keras.utils import plot_model
from tester_ui import tester_ui

def load_compl_model(model_path, img_shape=(None,None,3)):
    complnet_inp = Input(shape=img_shape, name='complnet_inp')
    complnet_out = completion_net(img_shape)(complnet_inp)
    compl_model = Model([complnet_inp], complnet_out)
    compl_model.load_weights(model_path, by_name=True)

    compl_model.summary()
    plot_model(compl_model, to_file='C_model_test.png', 
               show_shapes=True)
    return compl_model

def load_image(img_path):
    origin = cv2.imread(img_path)
    origin = cv2.cvtColor(origin,cv2.COLOR_BGR2RGB)
    origin = origin.astype(np.float32) / 255
    return origin, origin.shape[:2]

def mask_from_user(mask_hw, origin):
    h,w = mask_hw
    print('-------- ui start! --------')
    bgr_origin = cv2.cvtColor(origin,cv2.COLOR_RGB2BGR)
    mask = tester_ui(bgr_origin)
    mean_mask = mask * np.mean(origin) # images 2
    #cv2.imshow('mask',mask); cv2.waitKey(0)
    #cv2.imshow('mean mask',mean_mask); cv2.waitKey(0)
    return mean_mask, np.logical_not(mean_mask).astype(np.float32)
    
def adjusted_image(image, shape, pad_value=0): # tested on only grayscale image.
    h,w,_ = image.shape

    d_h = shape[0] - h
    if d_h > 0:
        d_top = d_h // 2
        d_bot = d_h - d_top
        image = np.pad(image, [(d_top,d_bot),(0,0),(0,0)], 
                       mode='constant', constant_values=pad_value)
        #print('+ y',image.shape)
    else:
        d_top = abs(d_h) // 2
        d_bot = abs(d_h) - d_top
        image = image[d_top:h-d_bot,:]
        #print('- y',image.shape)

    d_w = shape[1] - w
    if d_w > 0:
        d_left = d_w // 2
        d_right = d_w - d_left
        image = np.pad(image, [(0,0),(d_left,d_right),(0,0)],
                       mode='constant', constant_values=pad_value)
        #print('+ x',image.shape)
    else:
        d_left = abs(d_w) // 2
        d_right = abs(d_w) - d_left
        image = image[:,d_left:w-d_right]
        #print('- x',image.shape)
    return image
    
def padding_removed(padded_img, no_pad_shape):
    h,w,_ = no_pad_shape
    ret_img = adjusted_image(padded_img, no_pad_shape)
    ret_img = np.pad(ret_img[1:], [(0,1), (0,0), (0,0)], mode='constant') # move up
    #ret_img = np.pad(ret_img[:-1], [(1,0), (0,0), (0,0)], mode='constant') # move down
    #ret_img = np.pad(ret_img[:,:-3], [(0,0), (3,0), (0,0)], mode='constant') # move right
    ret_img = np.pad(ret_img[:,2:], [(0,0), (0,2), (0,0)], mode='constant') # move left
    return ret_img

import sys
imgpath = sys.argv[1]
origin, hw = load_image(imgpath)
#mean_mask, not_mask = mask_from_user(hw, origin)
mean_mask, not_mask = np.load('mean_mask.npy'), np.load('not_mask.npy')
holed_origin = origin * not_mask
complnet_input = np.copy(holed_origin) + mean_mask

h,w = hw
complnet_input = complnet_input[:,:,0]
complnet_input = complnet_input.reshape((1,h,w,1))

#compl_model = load_compl_model('./output/complnet_1099.h5', (None,None,1))
compl_model = load_compl_model('./output/old/192x_200e_complnet_199.h5', (None,None,1))
complnet_output = compl_model.predict(
                    [complnet_input.reshape((1,h,w,1))]
                  )
complnet_output = complnet_output.reshape(
                    complnet_output.shape[1:]
                  )
complnet_output = padding_removed(complnet_output, origin.shape)

mask = np.logical_not(not_mask).astype(np.float32)
completed = complnet_output * mask + holed_origin

print(complnet_output.shape)

bgr_origin = cv2.cvtColor(origin,cv2.COLOR_RGB2BGR)
cv2.imshow('origin',bgr_origin); cv2.waitKey(0)
#cv2.imshow('mean_mask',mean_mask); cv2.waitKey(0)
#cv2.imshow('not_mask',not_mask); cv2.waitKey(0)
#cv2.imshow('mask',mask); cv2.waitKey(0)
#cv2.imshow('holed_origin',holed_origin); cv2.waitKey(0)
#cv2.imshow('complnet_input',complnet_input); cv2.waitKey(0)
#cv2.imshow('complnet_output',complnet_output); cv2.waitKey(0)
bgr_completed = cv2.cvtColor(completed,cv2.COLOR_RGB2BGR)
cv2.imshow('completed',bgr_completed); cv2.waitKey(0)

#np.save('mean_mask',mean_mask)
#np.save('not_mask',not_mask)
print('is it ok?')
