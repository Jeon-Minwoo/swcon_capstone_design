import cv2
from cv2 import circle
import numpy as np
import math

def make_white_bg():
    white_bg = np.full((1600,2560,3), 255, dtype=np.uint8)
    cv2.imshow('white_bg', white_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('./img/white_bg.png', white_bg)
    
def make_circle():
    black_circle_pattern = np.full((1600, 2560, 3), 255, dtype=np.uint8)
    black_circle_pattern = cv2.circle(black_circle_pattern, (1280, 800), 50, (0,0,0), -1)
    cv2.imshow('black_circle_pattern', black_circle_pattern)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./img/black_circle_pattern.jpg', black_circle_pattern)

def make_calibration_bg():
    cal_bg = np.full((1600,2560,3), 255, dtype=np.uint8)
    x = 1280; y = 800
    l = [200, 400, 600]
    cal_bg = cv2.circle(cal_bg, (1280,800), 10, (0,0,0),-1)
    for i in l:
        cal_bg = cv2.line(cal_bg,(x-i, y), (x, y+i), (0,0,0), 4, cv2.LINE_AA)
        cal_bg = cv2.line(cal_bg,(x, y+i), (x+i, y), (0,0,0), 4, cv2.LINE_AA)
        cal_bg = cv2.line(cal_bg,(x+i, y), (x, y-i), (0,0,0), 4, cv2.LINE_AA)
        cal_bg = cv2.line(cal_bg,(x, y-i), (x-i, y), (0,0,0), 4, cv2.LINE_AA)
        
    cv2.imshow('cal_bg', cal_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./img/cal_bg.png', cal_bg)
        
def make_circle_pattern():
    x = 1280; y = 800
    circle_pattern = np.full((1600, 2560, 3), 255, dtype=np.uint8)
    for i in range(600, 0, -120):
        circle_pattern = cv2.circle(circle_pattern, (1280, 800), i, (0,0,0), 60)
        
    #circle_pattern = cv2.circle(circle_pattern, (1280, 800), 5, (0,0,0), -1)
    cv2.imshow('cir_pat', circle_pattern)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./img/cir_pat_2_5.png', circle_pattern)




def make_gaussian_circle():
    gaussian_circle_pattern = np.full((1080, 1920,1), 255, dtype=np.uint8)
    #cv2.imshow('gaussian_circle_pattern', gaussian_circle_pattern)
    kernel2d = np.zeros((150, 150, 1))
    kernel1d = cv2.getGaussianKernel(150, -1)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    print(kernel2d)
    kernel_norm = cv2.normalize(kernel2d, None, 0, 255, cv2.NORM_MINMAX)
    kernel_norm = kernel_norm.astype(np.uint8)
    kernel_norm = 255 - kernel_norm
    kernel_norm = np.reshape(kernel_norm,(150,150,1))
    gaussian_circle_pattern[465:465+150,885:885+150] = kernel_norm
    
    cv2.imshow('gaussian_circle_pattern', gaussian_circle_pattern)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('./img/black_gaussian_circle_pattern.jpg', gaussian_circle_pattern)

def make_cross_pattern():
    cross_pattern = np.full((1600, 2560, 3), 255, dtype=np.uint8)
    cross_pattern = cv2.line(cross_pattern, (0,800), (2560, 800), (0,0,0), 2, cv2.LINE_AA)
    cross_pattern = cv2.line(cross_pattern, (1280,0), (1280, 1600), (0,0,0), 2, cv2.LINE_AA)
    cv2.imshow('cross_pattern', cross_pattern)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./img/cross_pattern_1.jpg', cross_pattern)

def make_line_circle():
    line_circle = np.full((1600, 2560, 3), 255, dtype=np.uint8)
    # 15도 간격으로 선 
    r = 500; pi = math.pi
    x = 1280; y = 800
    print(round(r*math.sin(6*pi/12)))
    for i in range(6, -6, -1):
        cir_x = round(r*math.cos(i*pi/12))
        cir_y = round(r*math.sin(i*pi/12))
        line_circle = cv2.line(line_circle, (x+cir_x, y+cir_y), 
                                 (x-cir_x, y-cir_y), (0,0,0), 1, cv2.LINE_AA)
    cv2.imshow('line_circle', line_circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./img/line_circle_1.jpg', line_circle)

def make_grid():
    grid = np.full((1600,2560,3), 255, dtype=np.uint8)
    for i in range(0, 1600, 240):
        grid = cv2.line(grid, (0, i), (2560, i), (0,0,0), 5, cv2.LINE_AA)
    for j in range(0, 2560, 240):
        grid = cv2.line(grid, (j, 0), (j, 1600), (0,0,0), 5, cv2.LINE_AA)
    
    cv2.imshow('grid_bg', grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./img/grid_bg_5_5.png', grid)
    
# def make_red_bg():
#     red_bg = np.full((1080, 1920, 3), 0, dtype=np.uint8)
#     red_bg[:,:,2] = 255
#     cv2.imshow('red_bg', red_bg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('./img/red_bg.jpg', red_bg)
    
# def make_green_bg():
#     green_bg = np.full((1080, 1920, 3), 0, dtype=np.uint8)
#     green_bg[:,:,1] = 255
#     cv2.imshow('green_bg', green_bg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('./img/green_bg.jpg', green_bg)
    
# def make_blue_bg():
#     blue_bg = np.full((1080, 1920, 3), 0, dtype=np.uint8)
#     blue_bg[:,:,0] = 255
#     cv2.imshow('blue_bg', blue_bg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('./img/blue_bg.jpg', blue_bg)
    
    
if __name__ == "__main__":
    make_white_bg()
    #make_circle()
    #make_gaussian_circle()
    #make_cross_pattern()
    #make_line_circle()
    # make_red_bg()
    # make_green_bg()
    # make_blue_bg()
    make_grid()
    make_calibration_bg()
    make_circle_pattern()