from keras.layers import Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential, Model

def completion_net(input_shape=(256, 256, 3),name='complnet'):
    """
    Architecture of the image completion network
    """
    model = Sequential(name=name)
    # default strides = (1,1)
    #--------------------------------
    model.add(Conv2D( 64, (5,5), padding='same',
                     input_shape=(None,None,3)));                   model.add(BatchNormalization()); model.add(Activation('relu'));
    #----------------
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'));   model.add(BatchNormalization()); model.add(Activation('relu'));
    model.add(Conv2D(128, (3,3), padding='same'));                  model.add(BatchNormalization()); model.add(Activation('relu'));
    #--------
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'));   model.add(BatchNormalization()); model.add(Activation('relu'));
    model.add(Conv2D(256, (3,3), padding='same'));                  model.add(BatchNormalization()); model.add(Activation('relu'));
    model.add(Conv2D(256, (3,3), padding='same'));                  model.add(BatchNormalization()); model.add(Activation('relu'));
    # dilation
    model.add(Conv2D(256, (3,3), dilation_rate=(2,2), padding='same'));     model.add(BatchNormalization()); model.add(Activation('relu'));
    model.add(Conv2D(256, (3,3), dilation_rate=(4,4), padding='same'));     model.add(BatchNormalization()); model.add(Activation('relu'));
    model.add(Conv2D(256, (3,3), dilation_rate=(8,8), padding='same'));     model.add(BatchNormalization()); model.add(Activation('relu'));
    model.add(Conv2D(256, (3,3), dilation_rate=(16,16), padding='same'));   model.add(BatchNormalization()); model.add(Activation('relu'));
    #--------
    model.add(Conv2D(256, (3,3), padding='same'));                          model.add(BatchNormalization()); model.add(Activation('relu'));
    model.add(Conv2D(256, (3,3), padding='same'));                          model.add(BatchNormalization()); model.add(Activation('relu'));
    #----------------
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'));  model.add(BatchNormalization()); model.add(Activation('relu'));
    model.add(Conv2D(128, (3,3), padding='same'));                          model.add(BatchNormalization()); model.add(Activation('relu'));
    #--------------------------------
    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'));   model.add(BatchNormalization()); model.add(Activation('relu'));
    model.add(Conv2D( 32, (3,3), padding='same'));                          model.add(BatchNormalization()); model.add(Activation('relu'));
    model.add(Conv2D(  3, (3,3), padding='same')); # no BN!
    model.add(Activation('sigmoid')); # or sigmoid?
    return model


def discrimination_net(global_shape=(128, 128, 3),
                       local_shape=(64, 64, 3),name='discrimnet'):
    g_img = Input(shape=global_shape)
    # 32x32 / 178x178 / 128
    x_g = Conv2D(64, (5,5), strides=(2,2), padding='same')(g_img)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    # 16x16 / 89x89 / 64
    x_g = Conv2D(128, (5,5), strides=(2,2), padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    # 8x8 / 44x44 / 32
    x_g = Conv2D(256, (5,5), strides=(2,2), padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    '''
    # 4x4 / 22x22 / 16
    x_g = Conv2D(512, (5,5), strides=(2,2), padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    # 4x4 / 11x11 / 8
    x_g = Conv2D(512, (5,5), strides=(2,2), padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    # 4x4 / 5x5 / 4
    '''
    # FC: 1024
    x_g = Flatten()(x_g)
    x_g = Dense(1024)(x_g) # then ReLU? or not?

    l_img = Input(shape=local_shape)
    # 16x16 / 89x89 / 64
    x_l = Conv2D(64, (5,5), strides=(2,2), padding='same')(l_img)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    # 8x8 / 44x44 / 32
    x_l = Conv2D(128, (5,5), strides=(2,2), padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    '''
    # 4x4 / 22x22 / 16
    x_l = Conv2D(256, (5,5), strides=(2,2), padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    # 4x4 / 11x11 / 8
    x_l = Conv2D(512, (5,5), strides=(2,2), padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    # 4x4 / 5x5 / 4
    '''
    # FC: 1024
    x_l = Flatten()(x_l)
    #x_l = Dense(1024)(x_l) # then ReLU? or not?
    x_l = Dense(512)(x_l) # then ReLU? or not?

    x = concatenate([x_g, x_l]) # no activation. or..?
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[g_img, l_img], outputs=x, name=name)

import numpy as np
import keras.backend as K
def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

if __name__ == "__main__":
    from keras.utils import plot_model
    complnet = completion_net()
    complnet.summary()
    plot_model(complnet, to_file='tmp_complnet.png', show_shapes=True)
    discrimnet = discrimination_net()
    discrimnet.summary()
    plot_model(discrimnet, to_file='tmp_discrimnet.png', show_shapes=True)

    print(get_model_memory_usage(6,complnet),'GB')
    print(get_model_memory_usage(6,discrimnet),'GB')
