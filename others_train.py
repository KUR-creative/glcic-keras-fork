import imghdr
import numpy as np
import cv2
import os
from keras.optimizers import Adadelta
from keras.layers import merge, Input, Lambda
from keras.models import Model
from keras.engine.topology import Container
import keras.backend as K
import matplotlib.pyplot as plt
from model import model_generator, model_discriminator

class DataGenerator(object):
    def __init__(self, image_size, local_size):
        self.image_size = image_size
        self.local_size = local_size
        self.reset()

    def reset(self):
        self.images = []
        self.points = []
        self.masks = []

    def flow_from_directory(self, root_dir, batch_size, hole_min=64, hole_max=128):
        img_file_list = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if imghdr.what(full_path) is None:
                    continue
                img_file_list.append(full_path)

        np.random.shuffle(img_file_list)
        for f in img_file_list:
            img = cv2.imread(f)
            img = cv2.resize(img, self.image_size)[:, :, ::-1]
            self.images.append(img)

            x1 = np.random.randint(0, self.image_size[0] - self.local_size[0] + 1)
            y1 = np.random.randint(0, self.image_size[1] - self.local_size[1] + 1)
            x2, y2 = np.array([x1, y1]) + np.array(self.local_size)
            self.points.append([x1, y1, x2, y2])

            w, h = np.random.randint(hole_min, hole_max, 2)
            p1 = x1 + np.random.randint(0, self.local_size[0] - w)
            q1 = y1 + np.random.randint(0, self.local_size[1] - h)
            p2 = p1 + w
            q2 = q1 + h

            m = np.zeros((self.image_size[0], self.image_size[1], 1), dtype=np.uint8)
            m[q1:q2 + 1, p1:p2 + 1] = 1
            self.masks.append(m)

            if len(self.images) == batch_size:
                inputs = np.asarray(self.images, dtype=np.float32) / 255
                points = np.asarray(self.points, dtype=np.int32)
                masks = np.asarray(self.masks, dtype=np.float32)
                self.reset()
                yield inputs, points, masks

def example_gan(result_dir="output", data_dir="data"):
    input_shape = (128, 128, 3)
    local_shape = (64, 64, 3)
    batch_size = 32
    n_epoch = 10

    #tc = int(n_epoch * 0.18)
    #td = int(n_epoch * 0.02)
    tc = 2
    td = 2
    alpha = 0.0004

    train_datagen = DataGenerator(input_shape[:2], local_shape[:2])

    generator = model_generator(input_shape)
    discriminator = model_discriminator(input_shape, local_shape)
    optimizer = Adadelta()

    # build model
    org_img = Input(shape=input_shape)
    mask = Input(shape=(input_shape[0], input_shape[1], 1))

    in_img = merge([org_img, mask],
                   mode=lambda x: x[0] * (1 - x[1]),
                   output_shape=input_shape)
    imitation = generator(in_img)
    completion = merge([imitation, org_img, mask],
                       mode=lambda x: x[0] * x[2] + x[1] * (1 - x[2]),
                       output_shape=input_shape)
    cmp_container = Container([org_img, mask], completion, name='g_container')
    cmp_out = cmp_container([org_img, mask])

    cmp_model = Model([org_img, mask], cmp_out)
    cmp_model.compile(loss='mse',
                      optimizer=optimizer)

    local_img = Input(shape=local_shape)
    d_container = Container([org_img, local_img], discriminator([org_img, local_img]),
                                                            name='d_container')
    d_model = Model([org_img, local_img], d_container([org_img, local_img]))
    d_model.compile(loss='binary_crossentropy', 
                    optimizer=optimizer)

    '''
    '''
    cmp_model.summary()
    d_model.summary()
    from keras.utils import plot_model
    plot_model(cmp_model, to_file='cmp_model.png', show_shapes=True)
    plot_model(d_model, to_file='d_model.png', show_shapes=True)
    def random_cropping(x, x1, y1, x2, y2):
        out = []
        for idx in range(batch_size):
            out.append(x[idx, y1[idx]:y2[idx], x1[idx]:x2[idx], :])
        return K.stack(out, axis=0)
    cropping = Lambda(random_cropping, output_shape=local_shape)

    for n in range(n_epoch):
        ''''''
        org_img = Input(shape=input_shape)
        mask = Input(shape=(input_shape[0], input_shape[1], 1))

        in_img = merge([org_img, mask],
                       mode=lambda x: x[0] * (1 - x[1]),
                       output_shape=input_shape)
        imitation = generator(in_img)
        completion = merge([imitation, org_img, mask],
                           mode=lambda x: x[0] * x[2] + x[1] * (1 - x[2]),
                           output_shape=input_shape)
        cmp_container = Container([org_img, mask], completion, name='g_container')
        cmp_out = cmp_container([org_img, mask])

        cmp_model = Model([org_img, mask], cmp_out)
        cmp_model.compile(loss='mse',
                          optimizer=optimizer)

        local_img = Input(shape=local_shape)
        d_container = Container([org_img, local_img], discriminator([org_img, local_img]),
                                                                name='d_container')
        d_model = Model([org_img, local_img], d_container([org_img, local_img]))
        d_model.compile(loss='binary_crossentropy', 
                        optimizer=optimizer)

        cmp_model = Model([org_img, mask], cmp_out)
        local_img = Input(shape=local_shape)
        d_container = Container([org_img, local_img], discriminator([org_img, local_img]),
                                                                name='d_container')
        d_model = Model([org_img, local_img], d_container([org_img, local_img]))
        '''
        for inputs, points, masks in train_datagen.flow_from_directory(data_dir, batch_size,
                                                                       hole_min=48, hole_max=64):
            cmp_image = cmp_model.predict([inputs, masks])
            local = []
            local_cmp = []
            for i in range(batch_size):
                x1, y1, x2, y2 = points[i]
                local.append(inputs[i][y1:y2, x1:x2, :])
                local_cmp.append(cmp_image[i][y1:y2, x1:x2, :])

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            g_loss = 0.0
            d_loss = 0.0
            if n < tc:
                g_loss = cmp_model.train_on_batch([inputs, masks], inputs)
                print("epoch: %d < %d [D loss: %e] [G mse: %e]" % (n,tc, d_loss, g_loss))
                
            else:
                #d_model.trainable = True
                d_loss_real = d_model.train_on_batch([inputs, np.array(local)], valid)
                d_loss_fake = d_model.train_on_batch([cmp_image, np.array(local_cmp)], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                print('train D',n,(tc+td),'|',d_loss,'|',g_loss)
                if n >= tc + td:
                    d_container.trainable = False
                    cropping.arguments = {'x1': points[:, 0], 'y1': points[:, 1],
                                          'x2': points[:, 2], 'y2': points[:, 3]}
                    all_model = Model([org_img, mask],
                                      [cmp_out, d_container([cmp_out, cropping(cmp_out)])])
                    all_model.compile(loss=['mse', 'binary_crossentropy'],
                                      loss_weights=[1.0, alpha], optimizer=optimizer)
                    g_loss = all_model.train_on_batch([inputs, masks],
                                                      [inputs, valid])
                #print("epoch: %d [D loss: %e] [G all: %e]" % (n, d_loss, g_loss))
                    print(all_model.metrics_names)
                    print('train ALL',n,'|',d_loss,'|',g_loss)

        '''
        if n < tc:
            print("epoch: %d < %d [D loss: %e] [G mse: %e]" % (n,tc, d_loss, g_loss))
        else:
            print('train D',n,(tc+td),'|',d_loss,'|',g_loss)
            if n >= tc + td:
                print(all_model.metrics_names)
                print('train ALL',n,'|',d_loss,'|',g_loss)
        '''


        num_img = min(5, batch_size)
        fig, axs = plt.subplots(num_img, 3)
        for i in range(num_img):
            axs[i, 0].imshow(inputs[i] * (1 - masks[i]))
            axs[i, 0].axis('off')
            axs[i, 0].set_title('Input')
            axs[i, 1].imshow(cmp_image[i])
            axs[i, 1].axis('off')
            axs[i, 1].set_title('Output')
            axs[i, 2].imshow(inputs[i])
            axs[i, 2].axis('off')
            axs[i, 2].set_title('Ground Truth')
        fig.savefig(os.path.join(result_dir, "result_%d.png" % n))
        plt.close()
        # save model
        generator.save(os.path.join(result_dir, "generator_%d.h5" % n))
        discriminator.save(os.path.join(result_dir, "discriminator_%d.h5" % n))

        K.clear_session()

def main():
    example_gan()

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
