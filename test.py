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
compl_model.load_weights('./output/complnet_12.h5',by_name=True)

compl_model.summary()
plot_model(compl_model, to_file='C_model_test.png', show_shapes=True)
