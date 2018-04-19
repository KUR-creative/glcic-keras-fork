import numpy as np
import h5py, os, cv2
from data_utils import ElapsedTimer


def imgpaths(img_dir):
    it = os.walk(img_dir)
    next(it)
    for root,dirs,files in it:
        for path in map(lambda name:os.path.join(root,name), files):
            yield path

'''
'''
def batch_generator(array, batch_size):
    ''' return centered origin image '''
    arr_len = array.shape[0] // batch_size # never use remainder..
    #np.random.shuffle(array) #TODO: is shuffle needed?? maybe.. it's stochastic!
    for idx in range(0,arr_len, batch_size):
        print(idx, idx+batch_size)
        yield array[idx:idx+batch_size].astype(np.float32) / 255


'''
if __name__ == "__main__":
    arr = []
    for imgpath in imgpaths('./data/'):
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        arr.append(img)
    arr = np.array(arr) 

    # save
    with h5py.File('data.h5','w') as f:
        f.create_dataset('images',data=arr)
        mean = np.mean(arr)
        print(mean)
        f.create_dataset('mean_pixel_value', data=mean)

    # load
    with h5py.File('data.h5','r') as f:
        #print(f['dataset1'][:])
        print(f['mean_pixel_value'][()])

        #arr = np.arange(3600).reshape((12,10,10,3))
        bat_size = 16
        #batch_generator(arr,bat_size)

        print('f', f['images'].shape)
        print('mean', f['mean_pixel_value'][()])
        for batch in batch_generator(f['images'],bat_size):
            cv2.imshow('img',batch[0])
            cv2.waitKey(0)
            print(batch.shape)

'''
def iter_mean(prev_mean,prev_size, now_sum,now_size):
    total = prev_size + now_size
    return prev_mean*prev_size/total + now_sum/total

def path2img(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

from tqdm import tqdm
def chunk_generator(np_array,chk_size):
    length = len(np_array)
    for beg_idx in range(0,length, chk_size):
        yield np_array[beg_idx:beg_idx+chk_size]

if __name__ == "__main__":
    img_size = 128
    num_img_elems = (img_size**2)*3

    data_dir = './data128_half/'
    chk_size = 10000
    #chk_size = 130000
    #data_dir = './data/'
    #chk_size = 100
    # save
    with h5py.File('data128_half.h5','w') as f:
        get_imglist_timer = ElapsedTimer('--------> get_imglist') 
        #-----------------------------------------------------
        imgpath_list = list(imgpaths(data_dir))
        num_imgs = len(imgpath_list)
        print('number of images: ', num_imgs)
        #-----------------------------------------------------
        get_imglist_timer.elapsed_time() 

        save_h5_timer = ElapsedTimer('--------> save_h5') 
        #-----------------------------------------------------
        mean = 0
        f.create_dataset('images', (num_imgs,img_size,img_size,3))
                         #compression='lzf')
        for idx,imgpath in tqdm(
                             enumerate(
                               chunk_generator(imgpath_list, 
                                               chk_size)),
                             total=num_imgs//chk_size):
            beg_idx = idx*chk_size
            chunk = list(
                      map(path2img, 
                          imgpath_list[beg_idx:beg_idx+chk_size]))
            #print('got chunk',flush=True)
            mean = iter_mean(mean, beg_idx*num_img_elems,
                             np.sum(chunk), len(chunk)*num_img_elems)
            #print('got mean',flush=True)
            f['images'][beg_idx:beg_idx+chk_size] = chunk
            #print('chunk saved',flush=True)
        #-----------------------------------------------------
        save_h5_timer.elapsed_time() 

        #mean = np.mean(arr)
        print( 'saved num_imgs', num_imgs )
        print( 'saved mean = ', mean )
        f.create_dataset('mean_pixel_value', data=mean)
        #print( 'real mean = ', np.mean(f['images'][:]) )

    # load
    with h5py.File('data128_half.h5','r') as f:
        bat_size = 3 
        print()
        print('===== loaded =====')
        print('f', f['images'].shape)
        print('mean', f['mean_pixel_value'][()])
    '''
        for batch in batch_generator(f['images'],bat_size):
            cv2.imshow('img',batch[0]); cv2.waitKey(0)
            cv2.imshow('img1',batch[1]); cv2.waitKey(0)
            cv2.imshow('img2',batch[2]); cv2.waitKey(0)
            #print(batch.shape)
    '''
