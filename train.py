from networks import completion_net, discrimination_net
from keras.layers import Input, Add, Multiply, merge
from keras.models import Model
from keras.engine.topology import Container
from keras.optimizers import Adadelta, Adam
from data_utils import gen_batch,ElapsedTimer
from keras.utils import plot_model

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, h5py
np.set_printoptions(threshold=np.nan, linewidth=np.nan)


BATCH_SIZE = 16
IMG_SIZE = 128
LD_CROP_SIZE = IMG_SIZE // 2  # LD means Local Discrimnator
MAX_LEN = IMG_SIZE // 2
MIN_LEN = IMG_SIZE // 4
IMG_SHAPE = (IMG_SIZE,IMG_SIZE,3)
LD_CROP_SHAPE = (LD_CROP_SIZE,LD_CROP_SIZE,3)
MASK_SHAPE = (IMG_SIZE,IMG_SIZE,1)

def cropping(imgs_yxhws):
    def crop(img_yxhw):
        img,yxhw = img_yxhw
        y = yxhw[0]; x = yxhw[1]
        h = yxhw[2]; w = yxhw[3]
        return tf.image.crop_to_bounding_box(img, y,x, h,w)
    return tf.map_fn(crop, imgs_yxhws, dtype=tf.float32, infer_shape=False)

#def completion_model():
complnet_inp = Input(shape=IMG_SHAPE, name='complnet_inp')
masked_origins_inp = Input(shape=IMG_SHAPE, name='masked_origins_inp')
maskeds_inp = Input(shape=MASK_SHAPE, name='maskeds_inp')

complnet_out = completion_net(IMG_SHAPE)(complnet_inp)
merged_out = Add()([masked_origins_inp, 
                     Multiply()([complnet_out, 
                                 maskeds_inp])])
compl_model = Model([masked_origins_inp, 
                     complnet_inp, 
                     maskeds_inp], merged_out)
compl_model.compile(loss='mse', optimizer=Adadelta())

#def discrimination_model():
origins_inp = Input(shape=IMG_SHAPE, name='origins_inp')
crop_yxhw_inp = Input(shape=(4,), dtype=np.int32, name='yxhw_inp')
local_cropped = merge([origins_inp,crop_yxhw_inp], mode=cropping, 
                      output_shape=MASK_SHAPE, name='local_crop')
discrim_out = discrimination_net(IMG_SHAPE,
                                 LD_CROP_SHAPE)([origins_inp,
                                                 local_cropped])
discrim_model = Model([origins_inp,crop_yxhw_inp], discrim_out)
discrim_model.compile(loss='binary_crossentropy', 
                      optimizer=Adadelta(lr=0.01)) # good? lol
                      #optimizer=Adam(lr=0.000001))
discrim_model.summary()
plot_model(discrim_model, to_file='D_model.png', show_shapes=True)
                      
#def joint_model():
d_container = Container([origins_inp,crop_yxhw_inp], discrim_out,
                        name='D_container')
d_container.trainable = False
joint_model = Model([masked_origins_inp,complnet_inp,maskeds_inp,
                     crop_yxhw_inp],
                    [merged_out,
                     d_container([merged_out,crop_yxhw_inp])])
                    #merged_out,discrim_out)
alpha = 0.0004
joint_model.compile(loss=['mse', 'binary_crossentropy'],
                    loss_weights=[1.0, alpha], optimizer=Adadelta())
joint_model.summary()
plot_model(joint_model, to_file='joint_model.png', show_shapes=True)
timer = ElapsedTimer('Total Training')
#--------------------------------------------------------------------------------------
with h5py.File('./data128_half.h5','r+') as data_file:
    data_arr = data_file['images']
    mean_pixel_value = data_file['mean_pixel_value'][()] / 255

    #num_epoch = 200
    #tc = int(num_epoch * 0.18)
    #td = int(num_epoch * 0.02)
    num_epoch = 13
    tc = 2
    td = 1
    print('tc=',tc,' td=',td)

    valids = np.ones((BATCH_SIZE, 1))
    fakes = np.zeros((BATCH_SIZE, 1))
    for epoch in range(num_epoch):
        epoch_timer = ElapsedTimer('1 epoch training time')
        #--------------------------------------------------------------------------------------
        for batch in gen_batch(data_arr, BATCH_SIZE, IMG_SIZE, LD_CROP_SIZE,
                               MIN_LEN, MAX_LEN, mean_pixel_value):
            origins, complnet_inputs, masked_origins, maskeds, ld_crop_yxhws = batch

            #batch_timer = ElapsedTimer('1 batch training time')
            #--------------------------------------------------------------------------------------
            if epoch < tc:
                mse_loss = compl_model.train_on_batch([masked_origins, complnet_inputs, maskeds],
                                                      origins)
            else:
                completed = compl_model.predict([masked_origins, complnet_inputs, maskeds])
                d_loss_real = discrim_model.train_on_batch([origins,ld_crop_yxhws],valids)
                d_loss_fake = discrim_model.train_on_batch([completed,ld_crop_yxhws],fakes)
                bce_d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                if epoch >= tc + td:
                    joint_loss,mse,gan = joint_model.train_on_batch([masked_origins,complnet_inputs,maskeds,
                                                                     ld_crop_yxhws],
                                                                    [origins, valids])
            #--------------------------------------------------------------------------------------
            #batch_timer.elapsed_time()
        #--------------------------------------------------------------------------------------
        epoch_timer.elapsed_time()

        if epoch < tc:
            print('epoch %d: [C mse loss: %e]' % (epoch, mse_loss))
        else:
            print('epoch %d: [C mse loss: %e] [D bce loss: %e]' % (epoch, mse_loss, bce_d_loss))
            if epoch >= tc + td:
                print('epoch %d: [joint loss: %e | mse loss: %e, gan loss: %e]' % (epoch, joint_loss, mse,gan))

                if epoch % 4 == 0:
                    result_dir = 'output'
                    completed = compl_model.predict([masked_origins, complnet_inputs, maskeds])
                    np.save(os.path.join(result_dir,'I_O_GT__%d.npy' % epoch),
                            np.array([complnet_inputs,completed,origins]))
                            # save predicted image of last batch in epoch.
                    compl_model.save(os.path.join(result_dir, "complnet_%d.h5" % epoch))
                    discrim_model.save(os.path.join(result_dir, "discrimnet_%d.h5" % epoch))
#--------------------------------------------------------------------------------------
timer.elapsed_time()
'''
if __name__ == "__main__":
    timer = ElapsedTimer()
    main()
    timer.elapsed_time()

    #plot_model(compl_model, to_file='mse_model.png', show_shapes=True)
    #model.summary()
'''

