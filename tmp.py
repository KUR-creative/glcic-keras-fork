import cv2
import numpy as np
drawing = False # true if mouse is pressed
mode = False # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
magenta = (255,0,255)
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),magenta,-1)
            else:
                cv2.circle(img,(x,y),5,magenta,-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),magenta,-1)
        else:
            cv2.circle(img,(x,y),5,magenta,-1)

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
cv2.destroyAllWindows()
# copy image and then preprocess the copyed image: confirm no magenta!
# draw magenta mask.
'''
import cv2
import numpy as np
# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)
# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()


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
'''
