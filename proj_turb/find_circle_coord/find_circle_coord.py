import cv2
import numpy as np
import math

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = [x,y]
        print(clicked)
    
# dot image 및 꼭지점 원 생성 함수    
def make_dot_img(resol_x, resol_y, x_dot, y_dot):
    bg_img = np.full((resol_y, resol_x,3), 255, dtype=np.uint8)
    dot_img = bg_img.copy()
    cv2.circle(dot_img, (x_dot, y_dot), 20, (0, 0, 0), -1) #점찍기

    #꼭지점 원 생성
    r = 50
    s = 20
    cv2.circle(dot_img, (r+s,r+s)  , r, (255,0,0), -1) #left top blue circle
    cv2.circle(dot_img, (r+s,resol_y-r-s), r, (0,170,0), -1) #left bottom 
    cv2.circle(dot_img, (resol_x-r-s,r+s), r, (0,0,255), -1) #right top
    cv2.circle(dot_img, (resol_x-r-s,resol_y-r-s), r, (0,0,0), -1) #right bottom
    cv2.imshow('dot_img', dot_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return dot_img




# 사각형 꼭지점 찾는 함수
def find_corner(img):  #input은 휴대폰으로 촬영한 사진
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ = img.copy()
    img_2 = img.copy()
    
    _, imthres = cv2.threshold(img_gray, 100, 255, cv2.THRESH_TOZERO)
    imthres = cv2.GaussianBlur(imthres,(5,5), 0)
    imthres = cv2.bilateralFilter(imthres,9,75,75)
    imthres = cv2.edgePreservingFilter(imthres, flags=1, sigma_s=45, sigma_r=0.2)
    
    #edge detection
    img_edged = cv2.Canny(imthres, 75, 200)
    cv2.imshow('img_edged', img_edged)
    #Contour
    contour, _ = cv2.findContours(img_edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_2, contour, -1, (0,255,0))
    cv2.imshow('ctr', img_2)
    cv2.waitKey(0)
    
    #contour = sorted(contour, key=cv2.contourArea, reverse=True)[:5]

    pts = np.empty((1,2), dtype=np.float32)
    circle_contour = []
    
    for i in contour:
        epsilon = 0.005 * cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, epsilon, True)
        if len(approx) > 15:
            circle_contour.append(approx)

    circle_contour = sorted(circle_contour, key=cv2.contourArea, reverse=True)
    circle_cnt = 0
    prev_cx = 0
    prev_cy = 0
    for j in circle_contour:
        mmt = cv2.moments(j)
        cx = int(mmt['m10']/mmt['m00'])
        cy = int(mmt['m01']/mmt['m00'])
        if prev_cx == cx or prev_cy == cy:
            prev_cx = cx
            prev_cy = cy
            continue
        prev_cx = cx
        prev_cy = cy
        circle_cnt +=1
        pts = np.append(pts, np.array([[cx, cy]]), axis=0)
        cv2.drawContours(img_, [j], -1, (0,255,0), 10)
        if circle_cnt == 4:
            break
    cv2.imshow('con', img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    pts = np.delete(pts, [0,0], axis=0)
    print(pts)
    return pts
    
#Homograpnhy TF image //반환 tf img
def TF(resol_x, resol_y, pts, img): #img - 휴대폰 촬영 이미지
    pts = pts.astype(np.int32)
    topLeft = np.array([0, 0], dtype=np.float32)
    bottomLeft = np.array([0,0], dtype=np.float32)
    topRight = np.array([0,0], dtype=np.float32)
    bottomRight = np.array([0,0], dtype=np.float32)
    print(img[1198, 3119, :])
    print(img[301, 519, :])
    print(img[2300, 2529, :])
    print(img[1903, 203, :])
    
    for i in pts:
        if img[i[1], i[0], 0] <30  and img[i[1], i[0], 1] <30 and img[i[1], i[0], 2] < 30:
            bottomRight = i
        elif img[i[1], i[0], 0] >100: #blue detect
            topLeft = i
        elif img[i[1], i[0], 2] >100:
            topRight = i
        else:
            bottomLeft = i

    print(topLeft, bottomLeft, topRight, bottomRight)
    #변환 전 4개의 좌표
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
    width = resol_x - 140
    height = resol_y - 140
    
    pts2 = np.float32([[0,0], [width-1,0], [width-1, height-1], [0, height-1]])
    
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    TF_img = cv2.warpPerspective(img, mtrx, (width, height))
    cv2.imshow('TF_img', TF_img)
    cv2.setMouseCallback('TF_img', mouse_handler)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return TF_img

#convolve function
def detect_point(TF_img):
    
    TF_img_gray = cv2.cvtColor(TF_img, cv2.COLOR_BGR2GRAY)
    TF_img_gray = cv2.GaussianBlur(TF_img_gray, (3, 3), 0)
    _, th = cv2.threshold(TF_img_gray, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('th', th)
    TF_img_ = th[60:-60, 60:-60] #아웃라이어 제외
    circle_img = TF_img.copy()[60:-60, 60:-60, :]
    cv2.imshow('TF_img_', TF_img_)
    cv2.setMouseCallback('TF_img_', mouse_handler)
    cv2.waitKey(0)
    
    circles = cv2.HoughCircles(TF_img_, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=10)
    circles = np.uint16(np.around(circles))
    print(circles) 
    for i in circles[0,:]:
        cv2.circle(circle_img, (i[0],i[1]), i[2], (0, 255, 0),2)
    cv2.imshow('detect circle', circle_img)
    cv2.waitKey(0)
    x = circles[0,0,0] + 60 + 70
    y = circles[0,0,1] + 60 + 70
    return x, y

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
    
    labtop_dot = cv2.imread('./img/labtop_dot14.jpg')
    cv2.imshow('labtop_dot', labtop_dot)
    
    pts = find_corner(labtop_dot)
    TF_img = TF(resol_x, resol_y, pts, labtop_dot)
    #cv2.imshow('TF_img', TF_img)
    
    x, y = detect_point(TF_img)
    print("좌표: ", x, y)
    print("차이: ", calc_distance(x, y, x_dot, y_dot))
    
    cv2.setMouseCallback('con', mouse_handler)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 