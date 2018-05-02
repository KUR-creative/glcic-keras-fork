#from train import cropping
from data_utils import gen_batch
import numpy as np
import cv2
from networks import completion_net, discrimination_net
from keras.layers import Input, Add, Multiply, merge
from keras.models import Model
from keras.utils import plot_model

VAR_IMG_SHAPE = (None,None,3)
VAR_MASK_SHAPE = (None,None,1)

complnet_inp = Input(shape=VAR_IMG_SHAPE, name='complnet_inp')
masked_origins_inp = Input(shape=VAR_IMG_SHAPE, name='masked_origins_inp')
masks_inp = Input(shape=VAR_MASK_SHAPE, name='masks_inp')

complnet_out = completion_net(VAR_IMG_SHAPE)(complnet_inp)
merged_out = Add()([masked_origins_inp, 
                     Multiply()([complnet_out, 
                                 masks_inp])])
compl_model = Model([masked_origins_inp, 
                     complnet_inp, 
                     masks_inp], merged_out)
compl_model.load_weights('./output/old/complnet_199.h5',by_name=True)

#compl_model.summary()
#plot_model(compl_model, to_file='C_model_test.png', show_shapes=True)


origin = cv2.imread('./data/test_images/hwkqrxnfr8vy.jpg')
#origin = cv2.imread('./data/test_images/download.jpg')
#origin = cv2.imread('./data/test_images/149_0.png')
#origin = cv2.imread('./data/test_images/big_square.jpg')
origin = cv2.cvtColor(origin,cv2.COLOR_BGR2RGB)
origin = origin.astype(np.float32) / 255
hw = origin.shape[:2]
print(hw)

# mask
h,w = hw
mask = np.zeros((h,w,1), dtype=np.float32)
#mask[199:378,520:690] = 1.0 # hwkqrxnfr8vy
mask[10:50,20:60] = 1.0 # other

# masked origin
not_mask = np.logical_not(mask).astype(np.float32)
masked_origin = origin * not_mask

# complnet input
complnet_input = np.copy(masked_origin)
#cv2.imshow('input',origin); cv2.waitKey(0)
#cv2.imshow('not_mask',not_mask); cv2.waitKey(0)
#cv2.imshow('masked_origin',masked_origin); cv2.waitKey(0)
#cv2.imshow('complnet_input',complnet_input); cv2.waitKey(0)

completed = compl_model.predict([masked_origin.reshape((1,h,w,3)), 
                                 complnet_input.reshape((1,h,w,3)), 
                                 mask.reshape((1,h,w,1))])
cv2.imshow('completed',completed.reshape((h,w,3))); cv2.waitKey(0)
