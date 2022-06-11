import cv2
import numpy as np

def make_black_bg():
    black_bg = np.full((1080,1920,3), 0, dtype=np.uint8) ##삼성 노트북 fhd 해상도 이미지 생성
    cv2.imshow('black_bg', black_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('./img/black_bg.jpg', black_bg)
    
def make_circle():
    circle_pattern = np.full((1080, 1920, 3), 0, dtype=np.uint8)
    circle_pattern = cv2.circle(circle_pattern, (960, 540), 50, (255,255,255), -1)
    cv2.imshow('circle_pattern', circle_pattern)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./img/circle_pattern.jpg', circle_pattern)
    
def make_gaussian_circle():
    gaussian_circle_pattern = np.full((1080, 1920,1), 0, dtype=np.uint8)
    #cv2.imshow('gaussian_circle_pattern', gaussian_circle_pattern)
    kernel2d = np.zeros((100, 100, 1))
    kernel1d = cv2.getGaussianKernel(100, 21)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    print(kernel2d)
    kernel_norm = cv2.normalize(kernel2d, None, 0, 255, cv2.NORM_MINMAX)
    kernel_norm = kernel_norm.astype(np.uint8)
    kernel_norm = np.reshape(kernel_norm,(100,100,1))
    gaussian_circle_pattern[490:490+100,910:910+100] = kernel_norm
    
    cv2.imshow('gaussian_circle_pattern', gaussian_circle_pattern)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('./img/gaussian_circle_pattern.jpg', gaussian_circle_pattern)
     
    
if __name__ == "__main__":
    make_black_bg()
    make_circle()
    make_gaussian_circle()
    
