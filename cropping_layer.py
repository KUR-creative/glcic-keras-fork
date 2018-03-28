import numpy as np
import cv2

from keras.layers import Input, merge, Activation
from keras.models import Model
#import keras.backend as K
import tensorflow as tf

# cropping layer.
imgs = []
imgs.append(cv2.imread('./data/0301/2310301_0.png'))
imgs.append(cv2.imread('./data/0301/2312301_0.png'))
imgs.append(cv2.imread('./data/0301/2314301_0.png'))
cv2.imshow('1', imgs[0]); cv2.waitKey(0)
cv2.imshow('2', imgs[1]); cv2.waitKey(0)
cv2.imshow('3', imgs[2]); cv2.waitKey(0)

CROP_SIZE = 32
yxhws = np.array([[64,64, CROP_SIZE,CROP_SIZE], 
                  [ 6, 6, CROP_SIZE,CROP_SIZE], 
                  [24,24, CROP_SIZE,CROP_SIZE]]) # crop size must be SAME!
imgs = np.array(imgs)

#img_inp = Input(shape=img.shape, dtype='uint8')
img_inp = Input(shape=imgs[0].shape) # float
activated = Activation('sigmoid')(img_inp)
yxhw_inp = Input(shape=(4,), dtype='int32')

def cropping(imgs_yxhws):
    def crop(img_yxhw):
        img,yxhw = img_yxhw
        y = yxhw[0]; x = yxhw[1]
        h = yxhw[2]; w = yxhw[3]
        return tf.image.crop_to_bounding_box(img, y,x, h,w)
    return tf.map_fn(crop, imgs_yxhws, dtype=tf.float32, infer_shape=False)

merge_layer = merge([activated, yxhw_inp], 
                    mode=cropping, output_shape=(32,32,3))
model = Model(inputs=[img_inp, yxhw_inp], outputs=merge_layer)
model.summary()

imgs = imgs / 255 
y = model.predict([imgs, yxhws])

# TODO: not just 1 img but batch of imgs!

print(y.shape)
print(y[0].shape)
print(type(y))
cv2.imshow('y1', y[0]); cv2.waitKey(0)
cv2.imshow('y2', y[1]); cv2.waitKey(0)
cv2.imshow('y3', y[2]); cv2.waitKey(0)

# Make above multi!
