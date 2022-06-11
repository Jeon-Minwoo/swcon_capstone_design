import cv2
import numpy as np
import math

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = [x,y]
        print(clicked)
        

bg_img = np.full((900,1440,3), 255, dtype=np.uint8)

x_dot, y_dot = input("점의 x, y 좌표를 입력: ").split()
x_dot = int(x_dot)
y_dot = int(y_dot)
dot_img = bg_img.copy()
dot_img = cv2.line(dot_img, (x_dot,y_dot), (x_dot,y_dot), (0,0,0), 10)
#dot_img = bg_img.copy()
#dot_img[y_dot][x_dot][:] = 0 

#cv2.imshow('bg_img', bg_img)
cv2.imshow('dot_img', dot_img)


##edge detection
labtop_dot = cv2.imread('./img/labtop_dot.jpg')
labtop_dot2 = labtop_dot.copy()
labtop_dot_gray = cv2.cvtColor(labtop_dot, cv2.COLOR_BGR2GRAY)

ret, imthres = cv2.threshold(labtop_dot_gray, 100, 255, cv2.THRESH_BINARY)
imthres = cv2.GaussianBlur(imthres,(5,5), 0)
#imthres = cv2.bilateralFilter(imthres,9,75,75)
imthres = cv2.edgePreservingFilter(imthres, flags=1, sigma_s=45, sigma_r=0.2)


labtop_edged = cv2.Canny(imthres, 50,200)
cv2.imshow("th", imthres)
cv2.imshow("edged", labtop_edged)

##contour
contour, hierachy = cv2.findContours(labtop_edged.copy(), cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
contour = sorted(contour, key=cv2.contourArea, reverse=True)[:2]
#cv2.drawContours(labtop_dot2, contour, -1, (0,255,0), 4)

for i in contour:
    epsilon = 0.05 * cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, epsilon, True)
    
cv2.drawContours(labtop_dot2, [approx], -1, (0,255,0), 4)
cv2.imshow('con', labtop_dot2)
    
pts = np.zeros((4,2), dtype=np.float32)
for i in range(4):
    pts[i] = approx[i][0]
    
print(pts)
sm = np.sum(pts, axis=1)
diff = [-dif[0] for dif in np.diff(pts, axis=1)]
print(sm)
print(diff)

topLeft = pts[np.argmin(sm)]     # x+y가 가장 작은값 - 좌상
bottomRight = pts[np.argmax(sm)] # x+y가 가장 큰값 - 우하 
topRight = pts[np.argmax(diff)] # x-y가 가장 큰값 - 우상
bottomLeft = pts[np.argmin(diff)] # x-y가 가장 작은값 - 좌하

#변환 전 4개의 좌표
pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
width = 1440
height = 900
print(pts1)
#변환 후 4개의 좌표
pts2 = np.float32([[0,0], [width-1,0], [width-1, height-1], [0, height-1]])

#변환행렬 계산
mtrx = cv2.getPerspectiveTransform(pts1, pts2)
print(mtrx)
#원근변환 적용
scanned = cv2.warpPerspective(labtop_dot, mtrx, (width, height))
cv2.imshow('scanned', scanned)

#convolve function / 아웃라이어를 제외한 scanned 이미지 - scanned2
scanned_gray = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
scanned_gray = cv2.GaussianBlur(scanned_gray, (5, 5), 0)
ret, th = cv2.threshold(scanned_gray, 70, 255, cv2.THRESH_BINARY_INV)
#scanned2 = th
scanned2 = th[10:-10, 10:-10]  #(890 * 1430)
#print(scanned2.shape)
kernel = np.zeros((7,7))
kernel[2:-2, 2:-2] = 255


output = np.zeros((scanned2.shape[0]-len(kernel)+1, scanned2.shape[1]-len(kernel)+1))
cv2.imshow('scanned2', scanned2)
value_max = 0
print(scanned2.shape)
print(output.shape)
max_x = 0
max_y = 0
for x in range(0, scanned2.shape[1]-len(kernel)+1):
    for y in range(0, scanned2.shape[0]-len(kernel)+1):
        #print(scanned2[y: y+len(kernel), x: x+len(kernel)].shape)
        value = (kernel * scanned2[y: y+len(kernel), x: x+len(kernel)]).sum()
        output[y, x] = value
        if value_max <value:
            value_max = value
            max_x = x + 13
            max_y = y + 13

print(max_x, max_y)
distance = math.sqrt((x_dot - max_x)**2 + (y_dot - max_y)**2) 
print("차이: ", distance)

cv2.setMouseCallback('con', mouse_handler)
cv2.waitKey(0)
cv2.destroyAllWindows() 