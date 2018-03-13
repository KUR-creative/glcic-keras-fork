import numpy as np
from keras.layers import Reshape, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
import keras.backend as K

def model_generator(input_shape=(256, 256, 3)):
    """
    Architecture of the image completion network
    """
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, strides=1, padding='same',
                     dilation_rate=(1, 1), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, strides=2,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=2,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(4, 4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(8, 8)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(16, 16)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2,
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, kernel_size=4, strides=2,
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(3, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model


def model_discriminator(global_shape=(256, 256, 3), local_shape=(128, 128, 3)):
    g_img = Input(shape=global_shape)
    l_img = Input(shape=local_shape)

    # Local Discriminator
    x_l = Conv2D(64, kernel_size=5, strides=2, padding='same')(l_img)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(128, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(256, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Flatten()(x_l)
    x_l = Dense(1024, activation='relu')(x_l)

    # Global Discriminator
    x_g = Conv2D(64, kernel_size=5, strides=2, padding='same')(g_img)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(128, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(256, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Flatten()(x_g)
    x_g = Dense(1024, activation='relu')(x_g)

    x = concatenate([x_l, x_g])
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[g_img, l_img], outputs=x)

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
    generator = model_generator()
    generator.summary()
    plot_model(generator, to_file='generator.png', show_shapes=True)
    discriminator = model_discriminator()
    discriminator.summary()
    plot_model(discriminator, to_file='discriminator.png', show_shapes=True)

    print(get_model_memory_usage(96,generator),'GB')
    print(get_model_memory_usage(96,discriminator),'GB')
