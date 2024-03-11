import boto3
import botocore
from botocore.exceptions import ClientError
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import rasterio.mask
from rasterio.enums import Resampling
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
from skimage import io
import cv2
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import logging

def clean(indices):
    cleaned_indices = {}
    k = 0
    for i, (x,y) in enumerate(indices[:-1]):
        if (x,y) not in cleaned_indices:
            cleaned_indices[(x,y)] = k
            k +=1
    
    for k, v in cleaned_indices.items():
        indices[v] = list(k)

    indices = indices[:len(cleaned_indices)]
    indices = np.append(indices, indices[0])
    return indices.reshape(-1,2)

def update_geotiff(image_dataset, updated_image_array, updated_transform):
    out_meta = image_dataset.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": updated_image_array.shape[1],
        "width": updated_image_array.shape[2],
        "transform": updated_transform
    })
    memfile = rasterio.MemoryFile()
    with memfile.open(**out_meta) as dataset:
            dataset.write(updated_image_array)
            del  updated_image_array
    
    del out_meta
    del dataset
 
    return memfile 

def resample_to_fixed_size(image_dataset, size):
    # resample data to target shape
    resampled_data = image_dataset.read(
        out_shape=(
            image_dataset.count,
            int(size),
            int(size)
        ),
        resampling=Resampling.bilinear
    )

    # scale image transform
    resample_transform = image_dataset.transform * image_dataset.transform.scale(
        (image_dataset.width / resampled_data.shape[-1]),
        (image_dataset.height / resampled_data.shape[-2])
    )

    return update_geotiff(image_dataset, resampled_data, resample_transform)
def download_image(bucket, prefix, filename,file_path):
        s3 = boto3.resource('s3')
        try:
            s3.Bucket(bucket).download_file(os.path.join(prefix, filename), file_path)
            
        except botocore.exceptions.ClientError as e:
            print('image file is downloaded at ', bucket, prefix, file_path, filename)
            if e.response['Error']['Code'] == "404":
                logging.error(f"Download failed. Image  {filename} was not found in bucket {bucket}")
                raise
            else:
                raise Exception(f"An error occurred while downloading the image : {e}")
    
class RoofTopImage (Dataset):
    def __init__(self,
                 s3_image_bucket,
                 prefix, 
                 annotation_path, 
                 max_corner_points=256, 
                 img_size=320, 
                 batchsize = 0,
                 prediction = False, 
                 prediction_file_path=None,
                 load_type = 'train',
                 cache_data = 'True',
                 cache_path='../data'
                ):
        self.s3_image_bucket = s3_image_bucket
        self.prefix = prefix
        self.load_type = load_type
        self.cache_data = cache_data
        self.cache_path = cache_path

        self.annotation_path = annotation_path
        self.prediction = prediction
        self.img_annotations = COCO(self.annotation_path)
        self.segs = []
        self.batchsize = batchsize
        
        if self.prediction:
            print('Loading prediction data ...', prediction_file_path)
            prediction_file = json.loads(open(prediction_file_path).read())
            self.img_annotations = self.img_annotations.loadRes(prediction_file)
        
        self.cat_id = self.img_annotations.getCatIds()
        self.img_ids = self.img_annotations.getImgIds(catIds= self.cat_id)
        self.img_ids = random.sample(range(0, len(self.img_ids)), 10000)
        
        self.file = open('exclude_images.txt', 'r')
        self.bad_images = [int(l.strip()) for l in self.file.readlines()]
        for v in self.bad_images:
            if v in self.img_ids:
                self.img_ids.remove(v)
                         
        self.window_size = img_size
        self.corner_points = max_corner_points

        if self.cache_data:
            Path(os.path.join(self.cache_path, self.load_type)).mkdir(parents=True, exist_ok=True)
            
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, indx):
        idx = self.img_ids[indx]
        img = self.img_annotations.loadImgs(idx)[0]
        
        img_path = os.path.join(os.path.join(self.cache_path,self.load_type), img['file_name'])
        
        if not os.path.exists(img_path): # now get it from s3
            download_image(self.s3_image_bucket, self.prefix, img['file_name'], img_path)
        

        image = io.imread(img_path)
        orig_img_size = image.shape[0]
        image = resize(image, (self.window_size, self.window_size, 3), anti_aliasing=True, preserve_range=True)
        image = torch.from_numpy(image).to(torch.float32)
        image = image.permute(2,0,1) / 255.0
        
        
        ann_ids = self.img_annotations.getAnnIds(imgIds=idx)
        coco_annotations = self.img_annotations.loadAnns(ann_ids)
        

        num_objs = len(coco_annotations)
        random.shuffle(coco_annotations)

        corner_mask = np.zeros((self.window_size, self.window_size))
        mask = np.zeros((self.window_size, self.window_size))

        gt_index = {}
        k = 0
        gt_indices = []
        gt_permutation_matrix = np.zeros((self.corner_points, self.corner_points))
 
        for i in range(num_objs):
            skip = False
            corner_points =  np.flip(np.array(coco_annotations[i]['segmentation'][0]).reshape(-1,2),1)
            #indices = corner_points.round().astype('int64')
            point_pair = corner_points/ (orig_img_size/self.window_size)
            indices = point_pair.round().astype('int64').clip(0, self.window_size - 1)
            
            indices = clean(indices)
            if indices.shape[0]-1  < 3:
                continue

            if indices[0][0] != indices[-1][0] or indices[0][1] != indices[-1][1]:
                continue

            for x,y in indices[:-1]:
                if (x,y) not in gt_index:
                    gt_index[(x,y)] = k
                    k+=1
                else:
                    #seen before
                    skip = True
                    break
            if skip:
                continue
                
            
            corner_mask[indices[:-1][:,0], indices[:-1][:,1]] = 1
            gt_indices.append(indices.tolist())
            
            rle = cocomask.frPyObjects(coco_annotations[i]['segmentation'],self.window_size,self.window_size)
            m = cocomask.decode(rle).astype('float32')
            m = m.reshape((self.window_size, self.window_size))
            mask = mask + m.reshape((self.window_size, self.window_size))
        
        vertices = [v for polygon in gt_indices for v in polygon[:-1]]
        num_vertices=0
       
        for polygon in gt_indices: # take care of cases where gt_indices in empty
            n = len(polygon)
            num_vertices += n-1
            # iterate through each vertex and its corresponding adjacent vertex
            for i in range(n-1):
                v1 = polygon[i]
                v2 = polygon[(i + 1) % (n-1)]
                gt_permutation_matrix[gt_index[tuple(v1)]][gt_index[tuple(v2)]] = 1 
                
        #vertices = np.array(vertices)
        gt_permutation_matrix[range(num_vertices, self.corner_points), range(num_vertices, self.corner_points)] =1
        #image_idx = torch.tensor([idx])
       
        return image,torch.from_numpy(gt_permutation_matrix), np.array(vertices),gt_index,torch.from_numpy(corner_mask), torch.from_numpy(mask)
    
    
    # get the actual resized image
    '''
        raster_img=rasterio.open(img_path)
        memfile = resample_to_fixed_size(raster_img, self.window_size)
        dataset = memfile.open()
        image = torch.from_numpy(dataset.read()).to(torch.float32) / 255.0

        del dataset
        del raster_img 
        del img
        del memfile

    '''
    '''
    #load annotations: prediction or groudtruth
    annids = self.img_annotations.getAnnIds(imgIds=img['id'])
    anns = self.img_annotations.loadAnns(annids)

    for indx, ann in enumerate(anns):
        rle= cocomask.frPyObjects(ann['segmentation'],self.window_size,self.window_size)
        m = cocomask.decode(rle).astype('float32')
        if indx ==0:
            mask = m.reshape((self.window_size, self.window_size))

        else:
            mask = mask + m.reshape((self.window_size, self.window_size))

    #Henok: because there are cracks in some of the masks
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel)
    '''  