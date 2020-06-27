import cv2 as cv
import numpy as np

def edge(img):

    #高斯模糊,降低噪声
    blurred = cv.GaussianBlur(img,(3,3),0)

    #灰度图像
    gray=cv.cvtColor(blurred,cv.COLOR_RGB2GRAY)

    #图像梯度
    xgrad=cv.Sobel(gray,cv.CV_16SC1,1,0)
    ygrad=cv.Sobel(gray,cv.CV_16SC1,0,1)

    #计算边缘
    #50和150参数必须符合1：3或者1：2
    edge_output=cv.Canny(xgrad,ygrad,50,150)

    dst = cv.bitwise_and(img,img,mask=edge_output)
    return dst