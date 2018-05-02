from data_utils import gen_batch
import numpy as np
import cv2
from networks import completion_net, discrimination_net
from keras.layers import Input, Add, Multiply, merge
from keras.models import Model
from keras.utils import plot_model

VAR_IMG_SHAPE = (None,None,3)

complnet_inp = Input(shape=VAR_IMG_SHAPE, name='complnet_inp')

complnet_out = completion_net(VAR_IMG_SHAPE)(complnet_inp)
compl_model = Model([complnet_inp], complnet_out)
#compl_model.load_weights('./output/complnet_12.h5',by_name=True)
compl_model.load_weights('./output/old/complnet_199.h5',by_name=True)

compl_model.summary()
plot_model(compl_model, to_file='tmpC_model_test.png', show_shapes=True)


#origin = cv2.imread('./data/test_images/1100100.png')
#origin = cv2.imread('./data/test_images/cvcroped1099100.jpg')
origin = cv2.imread('./data/test_images/1099100.png')
#origin = cv2.imread('./data/test_images/hwkqrxnfr8vy.jpg')
#origin = cv2.imread('./data/test_images/download.jpg')
#origin = cv2.imread('./data/test_images/download 2.jpg')
#origin = cv2.imread('./data/test_images/images 2.jpg')
#origin = cv2.imread('./data/test_images/56478648_p0_master1200.jpg')
#origin = cv2.imread('./data/test_images/images 3.jpg')
#origin = cv2.imread('./data/test_images/149_0.png')
#origin = cv2.imread('./data/test_images/big_square.jpg')
origin = cv2.cvtColor(origin,cv2.COLOR_BGR2RGB)
origin = origin.astype(np.float32) / 255
hw = origin.shape[:2]

# mask
h,w = hw
mask = np.zeros((h,w,1), dtype=np.float32)
#mask[199:378,520:690] = 1.0 # hwkqrxnfr8vy
#mask[60:100,30:130] = 1.0 # other

# masked origin
not_mask = np.logical_not(mask).astype(np.float32)
masked_origin = origin * not_mask
masked_origin[199:378,520:690] = np.mean(origin)
masked_origin[60:100,30:130] = np.mean(origin)

completed = compl_model.predict([masked_origin.reshape((1,h,w,3))])
print(origin.shape)
print(completed.shape)
#cv2.imshow('input',origin); cv2.waitKey(0)
cv2.imshow('masked_origin',masked_origin); cv2.waitKey(0)
cv2.imshow('completed',completed.reshape((completed.shape[1],
                                          completed.shape[2],
                                          3))); cv2.waitKey(0)
