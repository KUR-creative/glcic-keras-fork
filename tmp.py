import cv2
import numpy as np
drawing = False # true if mouse is pressed
mode = False # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
color = (0,0,0)
# mouse callback function
def draw_mask(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(mask,(ix,iy),(x,y),color,-1)
            else:
                cv2.circle(mask,(x,y),5,color,-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(mask,(ix,iy),(x,y),color,-1)
        else:
            cv2.circle(mask,(x,y),5,color,-1)


def tester_ui(mean_pixel_value):
    bg = cv2.imread('./data/test_images/images 2.jpg')
    mask = np.ones(bg.shape, np.float32)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_mask)
    screen = np.empty(bg.shape)
    while(1):
        screen = (bg.astype(np.float32) / 255) * mask 
        cv2.imshow('image',screen)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 13: #ESC = 27, CarriageReturn(enter) = 13
            break
    cv2.destroyAllWindows()
    cv2.imshow('mask',np.logical_not(mask).astype(np.float32)); cv2.waitKey(0)

tester_ui(100)
