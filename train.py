from networks import completion_net
from keras.layers import Input, Add, Multiply
from keras.models import Model
from keras.optimizers import Adadelta
from data_utils import gen_batch
import matplotlib.pyplot as plt
import numpy as np
import os, h5py

BATCH_SIZE = 16
IMG_SIZE = 128
MAX_LEN = IMG_SIZE // 2
MIN_LEN = IMG_SIZE // 4


IMG_SHAPE = (IMG_SIZE,IMG_SIZE,3)
MASK_SHAPE = (IMG_SIZE,IMG_SIZE,1)
complnet_inp = Input(shape=IMG_SHAPE, name='complnet_inp')
masked_origins_inp = Input(shape=IMG_SHAPE, name='masked_origins_inp')
maskeds_inp = Input(shape=MASK_SHAPE, name='maskeds_inp')

complnet_out = completion_net(IMG_SHAPE)(complnet_inp)
merged_out = Add()([masked_origins_inp, 
                     Multiply()([complnet_out, 
                                 maskeds_inp])])
model = Model([masked_origins_inp, complnet_inp, maskeds_inp], 
              merged_out)
model.compile(loss='mse', optimizer=Adadelta())

with h5py.File('data.h5','r+') as data_file:
    data_arr = data_file['images']
    mean_pixel_value = data_file['mean_pixel_value'][()] / 255

    for epoch in range(100):
        for batch in gen_batch(data_arr, BATCH_SIZE, IMG_SIZE, 
                               MIN_LEN, MAX_LEN, mean_pixel_value):
            origins, complnet_inputs, masked_origins, maskeds = batch
            mse_loss = model.train_on_batch([masked_origins,
                                             complnet_inputs,
                                             maskeds],
                                            origins)
        print('epoch %d: [C mse loss: %e]' % (epoch, mse_loss))

        outputs = model.predict([masked_origins, 
                                 complnet_inputs, 
                                 maskeds])
        result_dir = 'output'
        np.save(os.path.join(result_dir,'I_O_GT__%d.npy' % epoch),
                np.array([complnet_inputs,outputs,origins]))

        model.save(os.path.join(result_dir, 
                                "complnet_%d.h5" % epoch))

'''
import time
class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

if __name__ == "__main__":
    timer = ElapsedTimer()
    main()
    timer.elapsed_time()
'''

if __name__ == "__main__":
    from keras.utils import plot_model
    plot_model(model, to_file='mse_model.png',show_shapes=True)
    model.summary()


