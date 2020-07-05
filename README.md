# Multi-Scale Progressive Fusion Network for Single Image Deraining (MSPFN)

This is an implementation of the MSPFN model proposed in the paper
([Multi-Scale Progressive Fusion Network for Single Image Deraining](https://arxiv.org/abs/2003.10985))
with TensorFlow.

# Requirements

- Python 3
- TensorFlow 1.1
- OpenCV
- tqdm
- glob
- sys

# Motivation

The repetitive samples of rain streaks in a rain image as well as its multi-scale versions 
(multi-scale pyramid images) may carry complementary information (e.g., similar appearance) 
to characterize target rain streaks. We explore the multi-scale representation 
from input image scales and deep neural network representations in a unified framework, 
and propose a multi-scale progressive fusion network (MSPFN) 
to exploit the correlated information of rain streaks across scales for single image deraining.

# Usage

## I. Train the MSPFN model

### Dataset Organization Form

If you prepare your own dataset, please follow the following form:
|--train_data  

    |--rainysamples  
        |--file1
                ：  
        |--file2
            :
        |--filen
        
    |--cleansamples
        |--file1
                ：  
        |--file2
            :
        |--filen
Then you can produce the corresponding '.npy' in the '/train_data/npy' file.
```
$ python preprocessing.py
```

### Training
Download training dataset ([Baidu Cloud](https://pan.baidu.com/s/18nurfhNbFt--Xzs5sAVvlA)), or prepare your own dataset like above form.

Run the following commands:
```
cd ./model
python train_MSPFN.py 
```

## II. Test the MSPFN model 

#### Quick Test
Download the pretrained models ([Baidu Cloud](https://pan.baidu.com/s/1gq16HTvJCHEXXc0V3t7lqw)).

Download the testing dataset (R100H, R100L, TEST100, TEST1200, TEST2800).

Run the following commands:
```
cd ./model/test
python test_MSPFN.py
```
The deraining results will be in './test/test_data/MSPFN'.

####  Test Your Own Dataset
Download the pretrained models.

Put your dataset in './test/test_data/'.

Run the following commands:
```
cd ./model/test
python test_MSPFN.py
```
The deraining results will be in './test/test_data/MSPFN'.

# Citation
```
@InProceedings{Kui_2020_CVPR,
	author = {Jiang, Kui and Wang, Zhongyuan and Yi, Peng and Huang, Baojin and Luo, Yimin and Ma, Jiayi and Jiang, Junjun},
	title = {Multi-Scale Progressive Fusion Network for Single Image Deraining},
	booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2020}
}
```