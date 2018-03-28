import numpy as np
np.set_printoptions(threshold=np.nan, linewidth=np.nan)
import cv2, utils

from keras.layers import Input, Dense, Flatten, merge, Merge, Activation, Add, Multiply
from keras.models import Model
#import keras.backend as K
import tensorflow as tf

'''
# input origin, generated, rand_mask
name = './data/0301/2310301_0.png'
img = cv2.imread(name)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow(name, img); cv2.waitKey(0)

oH,oW = img.shape[:2]
lH,lW = (oH//2,oW//2)
lH = lW = max(lH,lW)
max_len = lH
min_len = max_len * 1 // 4 
print(min_len)

masked, mask_yx, mask_hw = utils.random_masked(img.shape,
                                               min_len,max_len, 
                                               mask_val=1)
cv2.imshow('mask', masked); cv2.waitKey(0)
mY,mX = mask_yx
mH,mW = mask_hw
#255masked[mY:mY+mH, mX:mX+mW] = 1

#print('-----')
#print(masked[:,:,0])
#print('-----')
#print(masked[:,:,1])
#print('-----')
#print(masked[:,:,2])
#print('-----')

#print(masked.shape, img.shape)
#print(masked.dtype, img.dtype)
not_masked = np.logical_not(masked).astype(np.uint8)
cv2.imshow('masked', masked); cv2.waitKey(0)

complnet_input = not_masked*img
complnet_input[mY:mY+mH, mX:mX+mW] = 128
cv2.imshow('complnet_input', complnet_input); cv2.waitKey(0)

murged = not_masked*img + masked*img
cv2.imshow('murged', murged); cv2.waitKey(0)

#print(img[:,:,0])
#print(masked[:,:,0])
#print(murged[:,:,0])
'''

origins = [] 
origins.append(cv2.imread('./data/0300/1000300_0.png'))
origins.append(cv2.imread('./data/0300/1004300_1.png'))
origins.append(cv2.imread('./data/0300/1006300_0.png'))

generateds = []
generateds.append(cv2.imread('./data/0300/1002300_1.png'))
generateds.append(cv2.imread('./data/0300/1007300_0.png'))
generateds.append(cv2.imread('./data/0300/1008300_0.png'))

maskes = []
maskes.append(utils.random_masked((128,128,3), 30, 50)[0])
maskes.append(utils.random_masked((128,128,3), 30, 50)[0])
maskes.append(utils.random_masked((128,128,3), 30, 50)[0])

not_maskeds = []
not_maskeds.append( np.logical_not(maskes[0]).astype(np.uint8) ) 
not_maskeds.append( np.logical_not(maskes[1]).astype(np.uint8) )
not_maskeds.append( np.logical_not(maskes[2]).astype(np.uint8) )

masked_origins = [1,2,3]
masked_origins[0] = not_maskeds[0] * origins[0]
masked_origins[1] = not_maskeds[1] * origins[1]
masked_origins[2] = not_maskeds[2] * origins[2]

masked_origin = np.array(masked_origins)
generated = np.array(generateds)
mask = np.array(maskes)

masked_origin = masked_origin.astype(np.float32) / 255
generated = generated.astype(np.float32) / 255
mask = mask.astype(np.float32)
print(masked_origin.dtype,generated.dtype,mask.dtype)

cv2.imshow('org',masked_origin[0])
cv2.imshow('gen',generated[0])
cv2.imshow('mask',mask[0])
cv2.waitKey(0)

#masked_origin = masked_origin.reshape((1,128,128,3))
#generated = generated.reshape((1,128,128,3))
#mask = mask.reshape((1,128,128,3))

#a = Input(shape=masked_origin.shape,dtype='float32')
#b = Input(shape=generated.shape,dtype='float32')
#c = Input(shape=mask.shape,dtype='float32')
a = Input(shape=masked_origin[0].shape)
b = Input(shape=generated[0].shape)
c = Input(shape=mask[0].shape)

def gD_input(abc):
    a,b,c = abc
    return a + b * c
#merge_layer = merge([a,b,c], mode=gD_input, output_shape=(128,128,3))
t = Multiply()([b, c])
merge_layer = Add()([a, t])
#print(set())
model = Model(inputs=[a, b, c], outputs=merge_layer)
model.summary()

y = model.predict([masked_origin, generated, mask])
print(y.shape)
print(y[0].shape)
print(type(y))
cv2.imshow('y1', y[0]); cv2.waitKey(0)
cv2.imshow('y2', y[1]); cv2.waitKey(0)
cv2.imshow('y3', y[2]); cv2.waitKey(0)
# to batch!


