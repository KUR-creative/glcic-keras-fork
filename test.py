from data_utils import gen_batch
import numpy as np
import cv2
from networks import completion_net, discrimination_net
from keras.layers import Input, Add, Multiply, merge
from keras.models import Model
from keras.utils import plot_model

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
    mean_mask = np.zeros((h,w,1), dtype=np.float32)
    #mean_mask[199:378,520:690] = np.mean(origin) # hwkqrxnfr8vy
    #mean_mask[60:100,30:130] = np.mean(origin) # other
    #mean_mask[105:136,70:102] = np.mean(origin) # test1
    #mean_mask[343:437,225:327] = np.mean(origin) # test2
    #mean_mask[750:972,456:677] = np.mean(origin) # test3
    mean_mask[30:100,22:74] = np.mean(origin) # images 2
    return mean_mask, np.logical_not(mean_mask).astype(np.float32)

def padding_removed(padded_img,no_pad_shape):
    pH,pW,_ = padded_img.shape
    nH,nW,_ = no_pad_shape
    dH = pH - nH
    dW = pW - nW
    #print('ps',padded_img.shape)
    #print('ns',no_pad_shape)
    #print('dH,dW',dH,dW)
    #print('ret',padded_img[0:pH-dH,0:pW-dW].shape)
    # TODO: change this! it's temporary implementation!
    # TODO: 0~pH-dH is incorrect!
    return padded_img[0:pH-dH,0:pW-dW]

#origin, hw = load_image('./data/test_images/test1.png')
#origin, hw = load_image('./data/test_images/test2.png')
#origin, hw = load_image('./data/test_images/test3.png')
origin, hw = load_image('./data/test_images/images 2.jpg')
mean_mask, not_mask = mask_from_user(hw, origin)
holed_origin = origin * not_mask
complnet_input = np.copy(holed_origin) + mean_mask

compl_model = load_compl_model('./output/old/complnet_199.h5')
h,w = hw
complnet_output = compl_model.predict(
                    [complnet_input.reshape((1,h,w,3))]
                  )
complnet_output = complnet_output.reshape(
                    complnet_output.shape[1:]
                  )
complnet_output = padding_removed(complnet_output, origin.shape)

mask = np.logical_not(not_mask).astype(np.float32)
completed = complnet_output * mask + holed_origin


cv2.imshow('origin',origin); cv2.waitKey(0)
cv2.imshow('mean_mask',mean_mask); cv2.waitKey(0)
cv2.imshow('not_mask',not_mask); cv2.waitKey(0)
cv2.imshow('mask',mask); cv2.waitKey(0)
cv2.imshow('holed_origin',holed_origin); cv2.waitKey(0)
cv2.imshow('complnet_input',complnet_input); cv2.waitKey(0)
cv2.imshow('complnet_output',complnet_output); cv2.waitKey(0)
cv2.imshow('completed',completed); cv2.waitKey(0)

print('is it ok?')
