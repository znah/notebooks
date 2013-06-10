import numpy as np
import cv2

from common import Timer

def box(a, r):
    return cv2.boxFilter(a, cv2.CV_32F, (2*r+1, 2*r+1), normalize=True, borderType=cv2.BORDER_REFLECT_101)

def guided_filter(I, p, r, eps):
    mean_I = box(I, r)
    mean_p = box(p, r)
    mean_Ip = box(I*p, r)
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = box(I*I, r)
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = box(a, r)
    mean_b = box(b, r)
    
    q = mean_a * I + mean_b
    return q



def guided_filter_rgb(I, p, r, eps):
    assert I.shape[-1] == 3

    def box(a):
        return cv2.boxFilter(a, cv2.CV_32F, (2*r+1, 2*r+1), normalize=True, borderType=cv2.BORDER_REFLECT_101)

    mean_I = box(I)
    mean_p = box(p)
    mean_Ip = box(I*p[...,np.newaxis])
    cov_Ip = mean_Ip - mean_I * mean_p[...,np.newaxis]

    h, w, chn = I.shape
    var_I = np.zeros((h, w, 6), np.float32)
    for c, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        var_I[...,c] = box(I[...,i]*I[...,j]) - mean_I[...,i]*mean_I[...,j]

    



    #cov_Ip = mean_Ip - mean_I * mean_p
    
    #mean_II = box(I*I, r) / N
    #var_I = mean_II - mean_I * mean_I
    
    #a = cov_Ip / (var_I + eps)
    #b = mean_p - a * mean_I
    
    #mean_a = box(a, r) / N
    #mean_b = box(b, r) / N
    
    #q = mean_a * I + mean_b
    #return q



if __name__ == '__main__':
    img = np.float32(cv2.imread('data/03_left.bmp')) / 255.0
    cv2.imshow('img', img)

    with Timer('rgb'):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray[300:, 400:][:200, :200] = 1

        img1 = img.copy()
        for ch in xrange(3):
            img1[...,ch] = guided_filter_rgb(img, img[...,ch], 16, 0.1**2)

    cv2.imshow('t', img1)
    cv2.waitKey()