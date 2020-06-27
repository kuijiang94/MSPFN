import os
import glob
import shutil
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import requests
import tarfile


def preprocess():
    print('... loading data')
    dataset_dir = "train_data\\cleansamples\\"
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    path = "G:\\2019ICIP数据集\\crop\\input"## original datasets
    file = os.listdir(path)
    count =0
    num =0
    for item in file:
        image_path = os.path.join(path, item)
        print(image_path)
        img = Image.open(image_path)
        num =0
        for i in range(0,img.size[0]-96+1,96):
            for j in range(0,img.size[1]-96+1,96):
                IMG = img.crop([i,j,i+96,j+96])
                IMG.save(os.path.join(dataset_dir,'{}_{}_{}_{}.png'.format(count,num,i//96,j//96)))
                num+=1
        count+=1
        print(count)
def main():
    preprocess()


if __name__ == '__main__':
    main()

