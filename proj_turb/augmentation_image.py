import cv2
import numpy as np
import os

def verticalflip(img):
    origin = img.copy()
    verticalflip_img = cv2.flip(origin, 0)
    return verticalflip_img

def horizontalflip(img):
    origin = img.copy()
    horizontalflip_img = cv2.flip(origin, 1)
    return horizontalflip_img

def bothflip(img):
    origin = img.copy()
    bothflip_img = cv2.flip(origin, -1)
    return bothflip_img

def all_image_augmentation():
    image_list = [img for img in os.listdir('./eval')]
    image_list = sorted(image_list)
    print(image_list)
    for i in image_list:
        path = f'./eval/{i}'
        print(path)
        img = cv2.imread(path)
        verticalflip_img = verticalflip(img)
        horizontalflip_img = horizontalflip(img)
        bothflip_img = bothflip(img)
        cv2.imwrite(path[:20]+'-01.png', verticalflip_img)
        cv2.imwrite(path[:20]+'-02.png', horizontalflip_img)
        cv2.imwrite(path[:20]+'-03.png', bothflip_img)
 
if __name__ == '__main__':
    all_image_augmentation()