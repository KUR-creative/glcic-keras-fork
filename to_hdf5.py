import numpy as np
import h5py, os, cv2


def imgpaths(img_dir):
    it = os.walk(img_dir)
    next(it)
    for root,dirs,files in it:
        for path in map(lambda name:os.path.join(root,name), files):
            yield path

def batch_generator(array, batch_size):
    ''' return centered origin image '''
    arr_len = array.shape[0] // batch_size # never use remainder..
    #np.random.shuffle(array) #TODO: is shuffle needed?? maybe.. it's stochastic!
    for idx in range(0,arr_len, batch_size):
        print(idx, idx+batch_size)
        yield array[idx:idx+batch_size].astype(np.float32) / 255

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
