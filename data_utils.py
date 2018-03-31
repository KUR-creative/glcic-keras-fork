#import time
import numpy as np
from utils import *
import h5py, cv2, os
import matplotlib.pyplot as plt

def get_random_maskeds(batch_size, img_size, 
                       min_mask_len, max_mask_len):
    maskeds = np.empty((batch_size,img_size,img_size,1))
    mask_yxhws = []
    for idx in range(batch_size):
        _, mask_yx, mask_hw = random_masked(maskeds[idx], 
                                            min_mask_len,
                                            max_mask_len)
        y,x = mask_yx
        h,w = mask_hw
        mask_yxhws.append((y,x,h,w))
    return maskeds, mask_yxhws

def get_complnet_inputs(masked_origins,mask_yxhws, 
                        batch_size, mean_pixel_value):
    complnet_inputs = np.copy(masked_origins)
    for idx in range(batch_size):
        y,x,h,w = mask_yxhws[idx]
        complnet_inputs[idx][y:y+h,x:x+w] = mean_pixel_value
    return complnet_inputs

def gen_batch(data_arr, batch_size, img_size, ld_crop_size,
              min_mask_len, max_mask_len, mean_pixel_value):
    def _get_crop_yx(mask_yxhw_arr):
        mY,mX, mH,mW = mask_yxhw_arr
        y,x = get_ld_crop_yx((img_size,img_size),
                             (ld_crop_size,ld_crop_size),
                             (mY,mX), (mH,mW))
        return y,x
    ''' yield minibatches '''
    arr_len = data_arr.shape[0] // batch_size # never use remainders..
    np.random.shuffle(data_arr) #shuffle needed.
    # TODO: DO NOT shuffle file! just random picking!
    for idx in range(0,arr_len, batch_size):
        unpreprocessed_imgs = data_arr[idx:idx+batch_size]
        origins = unpreprocessed_imgs.astype(np.float32) / 255

        maskeds, mask_yxhws = get_random_maskeds(batch_size, 
                                                 img_size, 
                                                 min_mask_len, 
                                                 max_mask_len);

        not_maskeds = np.logical_not(maskeds).astype(np.float32)
        masked_origins = origins * not_maskeds

        complnet_inputs = get_complnet_inputs(masked_origins,
                                              mask_yxhws,
                                              batch_size,
                                              mean_pixel_value)
        #print(mask_yxhws)
        #print(ld_crop_size, type(maskeds))
        ld_crop_yxhws = np.empty((batch_size,4),dtype=int)
        for idx,(y,x) in enumerate(map(_get_crop_yx,mask_yxhws)):
            ld_crop_yxhws[idx] = y,x, ld_crop_size,ld_crop_size
        #print(ld_crop_yxhws)
        
        yield origins, complnet_inputs, masked_origins, maskeds, ld_crop_yxhws

def write_result_img(npy_path,img_path,batch_size,size):
    result = np.load(npy_path) 
    # size = image size
    result_img = np.empty((batch_size*size, 3*size, 3))
    for i in range(batch_size): 
        result_img[i*size:(i+1)*size, 0*size:1*size] = result[0,i]
        result_img[i*size:(i+1)*size, 1*size:2*size] = result[1,i]
        result_img[i*size:(i+1)*size, 2*size:3*size] = result[2,i]
    # convert correct image format
    result_img = (result_img * 255).astype(np.uint8)
    result_img = cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, result_img)

#TODO: load saved complnet and predict!
#TODO: create interactive demo!
if __name__ == "__main__":
    '''
    batch_size = 16
    img_size = 128
    maxl = img_size // 2
    minl = img_size // 4
    with h5py.File('data.h5','r+') as data_file: #TODO:'+' for shuffling.
        data_arr = data_file['images']
        mean_pixel_value = data_file['mean_pixel_value'][()] / 255

        for batch in gen_batch(data_arr, batch_size, 
                               img_size, img_size // 2, 
                               minl,maxl,mean_pixel_value):
            origins, complnet_inputs, masked_origins, maskeds, ld_crop_yxhws = batch
            lY,lX, lH,lW = ld_crop_yxhws[0]
            print('uwang good',ld_crop_yxhws)
            cv2.imshow('img',origins[0]); cv2.waitKey(0)
            #cv2.imshow('img2',origins[batch_size-1]); cv2.waitKey(0)
            cv2.imshow('ab',masked_origins[0]); cv2.waitKey(0) 
            #cv2.imshow('ab2',masked_origins[batch_size-1]); cv2.waitKey(0)
            cv2.imshow('complnet_inp',complnet_inputs[0]); cv2.waitKey(0)
            #cv2.imshow('complnet_inp2',complnet_inputs[batch_size-1]); cv2.waitKey(0) 
            cv2.imshow('ld_crop',complnet_inputs[0][lY:lY+lH,lX:lX+lW]); cv2.waitKey(0)
            #cv2.imshow('ld_crop2',complnet_inputs[batch_size-1][lY:lY+lH,lX:lX+lW]); cv2.waitKey(0)
    #print(result.shape) 
    '''
    bat_size = 16
    img_size = 128
    write_result_img('./output/I_O_GT__9.npy',
                     './output/result9.png',bat_size,img_size)
