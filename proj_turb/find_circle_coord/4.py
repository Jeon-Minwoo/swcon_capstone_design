import cv2
import numpy as np
import math

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = [x,y]
        print(clicked)
    
# dot image 및 꼭지점 원 생성 함수    
def make_dot_img(resol_x: int, resol_y: int, x_dot, y_dot):
    bg_img = np.full((resol_y, resol_x,3), 255, dtype=np.uint8)
    dot_img = bg_img.copy()
    dot_img = cv2.line(dot_img, (x_dot,y_dot), (x_dot,y_dot), (0,0,0), 4) #점찍기
    #모서리 기준 추가 이미지 (직사각형 기준)
    s = 100
    cv2.rectangle(dot_img, (0, 0), (s,s), (0,0,0), -1)  #좌상
    cv2.rectangle(dot_img, (0, resol_y-s), (s,resol_y), (255,0,0), -1) #좌하
    cv2.rectangle(dot_img, (resol_x-s, 0), (resol_x,s), (255,0,0), -1)
    cv2.rectangle(dot_img, (resol_x-s, resol_y-s), (resol_x,resol_y), (255,0,0), -1)

    
    cv2.imshow('dot_img', dot_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return dot_img


# 사각형 꼭지점 찾는 함수
def find_corner(img):  #input은 휴대폰으로 촬영한 사진
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ = img.copy()
    
    corners = cv2.goodFeaturesToTrack(img_gray, 10, 0.1, 500, blockSize=100)#, useHarrisDetector=True, k=0.06)
    corners= np.int32(corners)
    print(corners)
    for corner in corners:
        x, y = corner[0]
        cv2.circle(img_, (x, y), 10, (0,0,255), 3, cv2.LINE_AA)
    cv2.imshow('Corners', img_)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

    # pts = np.zeros((4,2), dtype=np.float32)
    # for i in range(4):
    #     pts[i] = approx[i][0]
    # print(pts)
    # return pts
    
#Homograpnhy TF image //반환 tf img
def TF(resol_x, resol_y, pts, img): #img - 휴대폰 촬영 이미지
    sm = np.sum(pts, axis=1)
    diff = [-dif[0] for dif in np.diff(pts, axis=1)]
    
    topLeft = pts[np.argmin(sm)]     # x+y가 가장 작은값 - 좌상
    bottomRight = pts[np.argmax(sm)] # x+y가 가장 큰값 - 우하 
    topRight = pts[np.argmax(diff)] # x-y가 가장 큰값 - 우상
    bottomLeft = pts[np.argmin(diff)] # x-y가 가장 작은값 - 좌하
    
    #변환 전 4개의 좌표
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
    width = resol_x
    height = resol_y
    
    pts2 = np.float32([[0,0], [width-1,0], [width-1, height-1], [0, height-1]])
    
    
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    TF_img = cv2.warpPerspective(img, mtrx, (width, height))
    #cv2.imshow('TF_img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return TF_img

#convolve function
def detect_point(TF_img):
    TF_img_gray = cv2.cvtColor(TF_img, cv2.COLOR_BGR2GRAY)
    TF_img_gray = cv2.GaussianBlur(TF_img_gray, (5, 5), 0)
    _, th = cv2.threshold(TF_img_gray, 70, 255, cv2.THRESH_BINARY_INV)
    
    TF_img_ = th[10:-10, 10:-10] #아웃라이어 제외
    kernel = np.zeros((7,7))
    kernel[2:-2, 2:-2] = 255
    output = np.zeros((TF_img_.shape[0]-len(kernel)+1, TF_img_.shape[1]-len(kernel)+1))
    value_max = 0
    max_x =0
    max_y =0
    
    for x in range(0, TF_img_.shape[1]-len(kernel)+1):
        for y in range(0, TF_img_.shape[0]-len(kernel)+1):
            #print(scanned2[y: y+len(kernel), x: x+len(kernel)].shape)
            value = (kernel * TF_img_[y: y+len(kernel), x: x+len(kernel)]).sum()
            output[y, x] = value
            if value_max <value:
                value_max = value
                max_x = x + 11
                max_y = y + 11 
    return max_x, max_y

def calc_distance(max_x, max_y, x_dot, y_dot):
    distance = math.sqrt((x_dot - max_x)**2 + (y_dot - max_y)**2) 
    return distance

    
if __name__ == "__main__":
    resol_x, resol_y = input("해상도 x y 입력: ").split()
    resol_x = int(resol_x)
    resol_y = int(resol_y)
    
    x_dot, y_dot = input("점의 x, y 좌표를 입력: ").split()
    x_dot = int(x_dot)
    y_dot = int(y_dot)
    
    dot_image = make_dot_img(resol_x, resol_y, x_dot, y_dot)
    cv2.imshow('dot_image', dot_image)
    
    labtop_dot = cv2.imread('./img/labtop_dot7.jpg')
    cv2.imshow('labtop_dot', labtop_dot)
    
    pts = find_corner(labtop_dot)
    TF_img = TF(resol_x, resol_y, pts, labtop_dot)
    cv2.imshow('TF_img', TF_img)
    
    x, y = detect_point(TF_img)
    print("좌표: ", x, y)
    print("차이: ", calc_distance(x, y, x_dot, y_dot))
    
    cv2.setMouseCallback('con', mouse_handler)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 