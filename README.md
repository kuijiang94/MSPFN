# Multi-Scale Progressive Fusion Network for Single Image Deraining (MSPFN)

This is an implementation of the MSPFN model proposed in the paper
([Multi-Scale Progressive Fusion Network for Single Image Deraining](https://arxiv.org/abs/2003.10985))
with TensorFlow.

# Requirements

- Python 3
- TensorFlow 1.12.0
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
        
    |--clean samples
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
Download training dataset ((raw images)[Baidu Cloud](https://pan.baidu.com/s/1usedYAf3gYOgAJJUDlrwWg), (**Password:4qnh**) (.npy)[Baidu Cloud](https://pan.baidu.com/s/1hOmO-xrZ2I6sI4lXiqhStA), (**Password:gd2s**)), or prepare your own dataset like above form.

Run the following commands:
```
cd ./model
python train_MSPFN.py 
```

## II. Test the MSPFN model 

#### Quick Test With the Raw Model (TEST_MSPFN_M17N1.PY)
Download the pretrained models ([Baidu Cloud](https://pan.baidu.com/s/1vfYbkbygiR4fC1I6eNcpmQ), (**Password:u5v6**)) ([Google Drive](https://drive.google.com/file/d/1nrjZtNs6AJYvfHi9TeCVTs50E57Fxgsc/view?usp=sharing)).

Download the commonly used testing rain dataset (R100H, R100L, TEST100, TEST1200, TEST2800) ([Google Drive](https://drive.google.com/file/d/1H6kigSTD0mucIoXOhpXZn3UqYytpS4TX/view?usp=sharing)), and the test samples and the labels of joint tasks form (BDD350, COCO350, BDD150) ([Baidu Cloud](https://pan.baidu.com/s/1xA4kgSyi9vZxVAeGRvc1tw), (**Password:0e7o**)). 
In addition, the test results of other competing models can be downloaded from here ([TEST1200, TEST100](https://drive.google.com/file/d/11nKUDRWJuapT8rogr6FARCMJF3rJoJtE/view?usp=sharing), [R100H, R100L](https://drive.google.com/file/d/1An5OChbJZnkhlbwGIDQ7wDh-xpkbELp9/view?usp=sharing)).

Run the following commands:
```
cd ./model/test
python test_MSPFN.py
```
The deraining results will be in './test/test_data/MSPFN'. We only provide the baseline for comparison. 
There exists the gap (0.1-0.2db) between the provided model and the reported values in the paper, which originates in the subsequent fine-tuning of hyperparameters, training processes and constraints.

####  Test the Retraining Model With Your Own Dataset (TEST_MSPFN.PY)
Download the pre-trained models.

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
	author = {Jiang, Kui and Wang, Zhongyuan and Yi, Peng and Chen, Chen and Huang, Baojin and Luo, Yimin and Ma, Jiayi and Jiang, Junjun},
	title = {Multi-Scale Progressive Fusion Network for Single Image Deraining},
	booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2020}
}
@ARTICLE{9294056,
  author={K. {Jiang} and Z. {Wang} and P. {Yi} and C. {Chen} and Z. {Han} and T. {Lu} and B. {Huang} and J. {Jiang}},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Decomposition Makes Better Rain Removal: An Improved Attention-guided Deraining Network}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2020.3044887}}
```
