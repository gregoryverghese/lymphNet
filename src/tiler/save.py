import os
import glob
import pickle
from PIL import Image

import lmdb
import numpy as np
from torch.utils.data import DataLoader, Dataset


class LMDBWrite():
    def __init__(self,db_path,map_size,write_frequency=10):
        self.db_path=db_path
        self.map_size=map_size
        self.env=lmdb.open(self.db_path, 
                           map_size=map_size,
                           writemap=True)
        self.write_frequency=write_frequency


    def __repr__(self):
        return f'LMBDWrite(size: {self.map_size}, path: {self.db_path})'
    

    def _serialize(self,image):
        image_bytes = image.tobytes()
        return image_bytes


    def write(self,image_paths): 
        txn=self.env.begin(write=True)
        for i, pth in enumerate(image_paths):
            image=np.array(Image.open(pth))
            value=self._serialize(image)
            key = f"{os.path.basename(pth)[:-4]}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
            if i % self.write_frequency == 0:
                #print("[%d/%d]" % (idx, len(data_loader)))
                txn.commit()
                txn = self.env.begin(write=True)
        txn.commit()
        self.env.close()


    def write_image(self,image,name): 
        txn=self.env.begin(write=True)
        value=self._serialize(image)
        key = f"{name}"
        txn.put(key.encode('ascii'), pickle.dumps(value))
        txn.commit()


    def close(self):
        self.env.close()



class LMDBRead():
    def __init__(self, db_path, image_size):
        self.db_path=db_path
        self.env=lmdb.open(self.db_path,readonly=True)
        self.image_size=image_size


    @property
    def num_keys(self):
        with self.env.begin() as txn:
            length = txn.stat()['entries']
        return length


    def __repr__(self):
        return f'LMDBRead(path: {self.db_path})'


    def get_keys(self):
        txn = self.env.begin()
        keys = [k for k, _ in txn.cursor()]
        #self.env.close()
        return keys


    def read_image(self,key):
        #env=lmdb.open(self.db_path,readonly=True)
        txn = self.env.begin()
        data = txn.get(key)
        image = pickle.loads(data)
        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape(self.image_size)
        #self.env.close()
        return image




class TFRecordWrite(self):
    def __init__(self, tf_path):
        self.tf_path=tf.path

   
    @staticmethod
    def _progress(self):
        pass

    def convert(self):
        
        if mask:
            data=zip(patch._patches, patch._masks)
        else:
            data=patch._patches
        with tf.io.TFRecordWriter(self.tf_path) as writer:
            for i,  in enumerate(data):
                printProgress(i,numImgs)
            mPath = os.path.dirname(m)
           
            m = os.path.join(mPath, os.path.basename(img[:-4]) + '_masks.png')
            maskName = os.path.basename(m)
            if not os.path.exists(m):
                check.append(maskName)
                continue
 
            maskName = os.path.basename(m)

            image = tf.keras.preprocessing.image.load_img(img)
            image = tf.keras.preprocessing.image.img_to_array(image,dtype=np.uint8)
            if stainNormalize=
            dims = image.shape
            image = tf.image.encode_png(image)
            
            mask = tf.keras.preprocessing.image.load_img(m)
            mask = tf.keras.preprocessing.image.img_to_array(mask, dtype=np.uint8)
            mask = tf.image.encode_png(mask)

            data = {
                'image': wrapBytes(image),
                'mask': wrapBytes(mask),
                'imageName': wrapBytes(os.path.basename(img)[:-4].encode('utf-8')),
                'maskName': wrapBytes(os.path.basename(m)[:-4].encode('utf-8')),
                'dims': wrapInt64(dims[0]) 
                }
               
            features = tf.train.Features(feature=data)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


























def convert(imageFiles, maskFiles, tfRecordPath, dim=None):
    '''
    load images and masks and serialize as a tfrecord file
    Args:
        imageFiles: imagefile paths
        maskFiles: maskfile paths
        tfRecordPath: path to save tfrecords
    '''

    numImgs = len(imageFiles)
    check=[]
    with tf.io.TFRecordWriter(tfRecordPath) as writer:
        for i, (img, m) in enumerate(zip(imageFiles, maskFiles)):
            printProgress(i,numImgs)
            imgName = os.path.basename(img)[:-4]
            maskName = os.path.basename(m)[:-10]
            mPath = os.path.dirname(m)
           
            m = os.path.join(mPath, os.path.basename(img[:-4]) + '_masks.png')
            maskName = os.path.basename(m)
            if not os.path.exists(m):
                check.append(maskName)
                continue
 
            maskName = os.path.basename(m)

            image = tf.keras.preprocessing.image.load_img(img)
            image = tf.keras.preprocessing.image.img_to_array(image,dtype=np.uint8)
            if stainNormalize=
            dims = image.shape
            image = tf.image.encode_png(image)
            
            mask = tf.keras.preprocessing.image.load_img(m)
            mask = tf.keras.preprocessing.image.img_to_array(mask, dtype=np.uint8)
            mask = tf.image.encode_png(mask)

            data = {
                'image': wrapBytes(image),
                'mask': wrapBytes(mask),
                'imageName': wrapBytes(os.path.basename(img)[:-4].encode('utf-8')),
                'maskName': wrapBytes(os.path.basename(m)[:-4].encode('utf-8')),
                'dims': wrapInt64(dims[0]) 
                }
               
            features = tf.train.Features(feature=data)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

