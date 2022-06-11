from re import S
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import yaml

from flask_eval_test import check_detection

def saturate_contrast(img, num):
    pic = img.copy()
    pic = pic.astype('int32')
    pic = np.clip(pic+(pic-128)*num, 0, 255)
    pic = pic.astype('uint8')
    return pic

def imshow_scale(img, s, title): #그냥 보기 편하게 하기 위한 메소드
    img_scale = cv.resize(img, dsize=(0, 0), fx=s, fy=s, interpolation=cv.INTER_LINEAR)
    cv.imshow(title, img_scale)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def clip_image_center(img_gray, s):
    gray_clip = np.zeros_like(img_gray)
    h, w = img_gray.shape
    hh = int(h/2)  ## 이미지 중점
    ww = int(w/2)

    gray_clip[hh-s:hh+s, ww-s:ww+s] = img_gray[hh-s:hh+s, ww-s:ww+s] ##ROI 설정
    imshow_scale(gray_clip, 0.4, 'gray')

    return gray_clip

def roi_detection_circle(img_src, im_gray):
    src = img_src.copy()
    src_ = img_src.copy()
    gray = cv.medianBlur(im_gray, 3)
    rows, cols = gray.shape
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.0, rows / 12, None, 20, 20, minRadius=250, maxRadius=350)
    #circles: N x 1 x 3 (x, y, radius)
    ww = int(cols/2)
    hh = int(rows/2)
    min_dist_center = cols * 100 #19200 그냥 큰값
    
    roi = np.ones(3) * -1
    
    if circles is not None: 
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(src_, (i[0], i[1]), i[2], (0,255,0),2 )
            dist_center = np.sqrt((i[0]-ww) * (i[0]-ww) + (i[1]-hh) * (i[1]-hh)) ##이미지 중점과 원의 거리
            if dist_center < min_dist_center:
                min_dist_center = dist_center ## 가장 이미지의 중심과 가까운 원의 중심 찾기
                if np.abs((i[0]-ww)) <100 and np.abs((i[1]-hh)) < 100: ## 가장 가까운 원의 중심일때 최소 중심과 차이 100미만
                    roi[0] = i[0]
                    roi[1] = i[1]
                    roi[2] = i[2]
            center = (i[0], i[1])
    
    roi = np.uint16(np.around(roi))
    cv.circle(src_, (roi[0], roi[1]), 3, (0, 0, 255), 3)
    cv.circle(src, (roi[0], roi[1]), roi[2], (255, 255, 255), 5)
    imshow_scale(src_, 0.4, 'all circle')
    imshow_scale(src, 0.4, 'circle roi')
    
    return roi

def masking_circle(img, roi_circle):
    img_mask = np.zeros_like(img)
    img_mask = cv.circle(img_mask, (roi_circle[0], roi_circle[1]), roi_circle[2], (255,255,255), -1)
    #bitwise_and 연산을 통한 마스킹..
    img_center = cv.bitwise_and(img.copy(), img_mask)
    return img_center    

def grid_calc(img_gray, roi_rect):
    gray = img_gray.copy()
    h, w = gray.shape
    
    grid_size = 80
    cx = (roi_rect[0] + roi_rect[2])/2 # 직사각형 좌표를 통해 원의 중점, 반지름 구하기
    cy = (roi_rect[1] + roi_rect[3])/2
    radius = (roi_rect[2] - roi_rect[0])/2
    
    list_mean = []
    list_std = []
    
    for r in range(roi_rect[1], roi_rect[3], grid_size): ##y
        for c in range(roi_rect[0], roi_rect[2], grid_size): ##x
            ## 원의 중점과 각 그리드 꼭지점 사이의 거리
            dist_lt = np.sqrt((c - cx)**2+(r - cy)**2) 
            dist_rt = np.sqrt((c + grid_size - cx) ** 2 + (r - cy) ** 2)
            dist_lb = np.sqrt((c - cx) ** 2 + (r + grid_size - cy) ** 2)
            dist_rb = np.sqrt((c + grid_size - cx) ** 2 + (r + grid_size - cy) ** 2)
            ## 각 그리드 꼭지점 까지의 거리 4가지 모두 원의 반지름보다 작은 그리드가 원 안의 그리드..!
            if dist_lt <= radius and dist_rt <= radius and dist_lb <= radius and dist_rb <= radius:
                patch = gray[r:r+grid_size, c:c+grid_size] ## 그리드 안의 모든 pixel들..
                mean = np.mean(patch)
                std = np.std(patch)
                list_mean = np.append(list_mean, mean)
                list_std = np.append(list_std, std)
                cv.rectangle(gray, (c, r), (c+grid_size, r+grid_size), (255,255,255), 3)
    
    cv.circle(gray, (int(cx), int(cy)), int(radius), (0,0,0),3)
    mmg = np.mean(np.array(list_mean))
    msg = np.mean(np.array(list_std))
    smg = np.std(np.array(list_mean))
    ssg = np.std(np.array(list_std))

    print('mean of mean of grid: %f'%mmg)
    print('mean of std of grid: %f'%msg)
    print('std of mean of grid: %f'%smg)
    print('std of std of grid: %f'%ssg)
    imshow_scale(gray, 0.4, 'grid')
    return mmg, msg, smg, ssg

def circle_segmentation(img_center, roi_circle):
    img_c = img_center.copy()
    im_rgb = cv.cvtColor(img_c, cv.COLOR_GRAY2BGR)
    #im_c = cv.GaussianBlur(img_c, (5,5),0)
    im_c = cv.medianBlur(img_c, 5)
    im_c = saturate_contrast(im_c, 1.0)
    imshow_scale(im_c, 0.4, 'contrastup')
    print('thresholding...')
    im_bin = cv.adaptiveThreshold(im_c, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY_INV, 151, 15)
    print('done...')
    im_bin = masking_circle(im_bin, roi_circle)

    print('draw particle...')
    #원을 둘러싼 직사각형 범위
    for r in range(roi_circle[1]-roi_circle[2], roi_circle[1]+roi_circle[2]):
        for c in range(roi_circle[0]-roi_circle[2], roi_circle[0]+roi_circle[2]):
            if im_bin[r,c] > 100: 
                cv.circle(im_rgb, (c,r), 3, (0, 255, 255), -1)
    print('done...')

    particle_sum = np.sum(im_bin/255) #어차피 이진 threshold라 0 or 255의 값을 가진다..
    print('particle sum: %d'%particle_sum)

    imshow_scale(im_bin, 0.4, 'bin')
    imshow_scale(im_rgb, 1.0, 'bin_overlap')

    return particle_sum

def cross_profile(img_center, roi_circle):
    img_gray = img_center.copy()
    img_gray = cv.medianBlur(img_gray, 5)
    img_rgb = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    
    # move circle center to image center
    h, w = img_gray.shape
    cx = int(w / 2)
    cy = int(h / 2)
    offset_x = cx - roi_circle[0]
    offset_y = cy - roi_circle[1]

    cv.circle(img_rgb, (cx, cy), 5, (255, 0, 0), 3)  ## blue - 이미지의 중점
    cv.circle(img_rgb, (roi_circle[0] , roi_circle[1]), 5, (0, 0, 255), 3) ## red - 원의 중점 // calibration이 잘 되지 않으면 차이가 발생할 수 밖에없다..
    imshow_scale(img_rgb, 0.4, 'cross')
    
    M = np.float64([[1, 0, offset_x], [0,1, offset_y]])
    img_gray_trans = cv.warpAffine(img_gray, M, (w, h)) ##2차원 이동 matrix 변환 - 원의 중점과 이미지의 중점 calibration
    img_gray_trans_ = img_gray_trans.copy()
    cv.circle(img_gray_trans_, (cx, cy), 5, (255, 0, 0), 3)
    imshow_scale(img_gray_trans_, 0.4, 'trans')
    
    cross_profile = np.array([])
    
    for i in range(0, 360, 30):
        R = cv.getRotationMatrix2D((cx, cy), i, 1)
        img_gray_rot =cv.warpAffine(img_gray_trans, R, (w, h))
        profile = img_gray_rot[cy, cx-roi_circle[2]+20:cx+roi_circle[2]-20]
        profile = np.reshape(profile, (1, -1))
        cross_profile = np.vstack([cross_profile, profile]) if cross_profile.size else profile
    #print(cross_profile.shape) (12, 632)

    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    for i in range(cross_profile.shape[0]):
        plt.plot(range(cross_profile.shape[1]), cross_profile[i,:])
    plt.ylim(0,255)
    
    cross_profile_mean = np.mean(cross_profile, axis=0)
    cross_profile_std = np.std(cross_profile, axis=0)
    cross_profile_area = 2* np.sum(cross_profile_std)
    
    curve_x = range(cross_profile.shape[1])
    curve_y = cross_profile_mean
    curve_fitted = np.polyfit(curve_x,curve_y,2)
    curve_a , curve_b, curve_c = curve_fitted  #ax^2 + bx + c = y
    print('curve coefficient: ', curve_fitted)
    curve_fit_y = curve_a * curve_x * curve_x + curve_b * curve_x + curve_c
    
    min_plot_x = np.argmin(curve_fit_y).astype(np.int16)
    min_plot_y = cross_profile_mean[min_plot_x].astype(np.int16)
    max_plot_x = 0
    max_plot_y = curve_c.astype(np.int16)
    
    # max_plot_x = np.argmax(cross_profile_mean).astype(np.int16)
    # max_plot_y = np.max(cross_profile_mean).astype(np.int16)
    # min_plot_x = np.argmin(cross_profile_mean).astype(np.int16)
    # min_plot_y = np.min(cross_profile_mean).astype(np.int16)
    gradient_plot = np.round(np.abs((max_plot_y - min_plot_y) / (max_plot_x - min_plot_x)),4)
    
    print(max_plot_x, max_plot_y, min_plot_x, min_plot_y, gradient_plot)
    
    print('std range:' ,cross_profile_area)
    
    plt.subplot(2,1,2)
    plt.plot(range(cross_profile.shape[1]), cross_profile_mean)
    plt.plot(curve_x, curve_fit_y)
    plt.fill_between(range(cross_profile.shape[1]), cross_profile_mean-cross_profile_std, cross_profile_mean+cross_profile_std, alpha=0.5)
    plt.ylim(0,255)
    
    plt.tight_layout()
    plt.show()    
    
    return curve_a, curve_b, curve_c, cross_profile_area, gradient_plot

def line_detection(img_gray_center, roi_circle):
    roi_circle = roi_circle.copy()
    roi_circle[2] = roi_circle[2] - 10
    img_gray_center = img_gray_center.copy()
    img_rgb_center = cv.cvtColor(img_gray_center, cv.COLOR_GRAY2BGR)
    gray = saturate_contrast(img_gray_center, 1.5)
    #gray = cv.GaussianBlur(gray, (3,3),0)
    imshow_scale(gray, 0.4, 'contrast_up_gray')
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 101, 25)
    th = masking_circle(th, roi_circle)
    imshow_scale(th, 0.4, 'th')
    edges = cv.Canny(th, 0, 100)
    imshow_scale(edges, 0.4, 'canny')
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=30, maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(img_rgb_center,(x1,y1),(x2,y2),(0,255,255),2)
    imshow_scale(img_rgb_center, 0.4, 'Hough')
    return img_rgb_center 

def crystal_detection(img_gray_center, roi_circle):
    roi_circle = roi_circle.copy()
    roi_circle[2] = roi_circle[2] - 10
    img_gray_center = img_gray_center.copy()
    img_gray_center = cv.medianBlur(img_gray_center, 3)
    img_rgb_center = cv.cvtColor(img_gray_center, cv.COLOR_GRAY2BGR)
    gray = saturate_contrast(img_gray_center, 1.3)
    imshow_scale(gray, 0.4, 'contrast_up_gray')
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 151, 10)
    th = masking_circle(th, roi_circle)
    imshow_scale(th, 0.4, 'th')
    edges = cv.Canny(th, 0, 100)
    imshow_scale(edges, 0.4, 'canny')
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #contours = contours[0]
    approx_contour = []
    for i in contours:
        epsilon = 0.001 * cv.arcLength(i, True)
        approx = cv.approxPolyDP(i, epsilon, True)
        if len(approx) > 5:
            approx_contour.append(approx)        

    for i in approx_contour:
        cv.drawContours(img_rgb_center, [i], -1, (0,255,255), 2)
        elipse = cv.fitEllipse(i)
        cv.ellipse(img_rgb_center, elipse, (255,0,0), 2)
        #cv.circle(img_rgb_center, (int(x), int(y)), int(radius), (255,0,0), 2)
    imshow_scale(img_rgb_center, 0.4, 'crystal')

def black_bg_line(img_gray_center, roi_circle):
    roi_circle[2] = roi_circle[2] - 10
    img_gray_center = img_gray_center.copy()
    img_rgb_center = cv.cvtColor(img_gray_center, cv.COLOR_GRAY2BGR)
    bk_bg_line = np.zeros_like(img_rgb_center)
    gray = saturate_contrast(img_gray_center, 1.5)
    gray = cv.GaussianBlur(gray, (3,3),0)
    #imshow_scale(gray, 0.4, 'contrast_up_gray')
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 101, 25)
    th = masking_circle(th, roi_circle)
    #imshow_scale(th, 0.4, 'th')
    edges = cv.Canny(th, 0, 100)
    #imshow_scale(edges, 0.4, 'canny')
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=30, maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(bk_bg_line,(x1,y1),(x2,y2),(0,255,255),2)
    #imshow_scale(bk_bg_line, 0.4, 'Hough')
    return bk_bg_line

def grid_detection(img_gray_center,bk_bg_grid,roi_circle):
    roi_circle = roi_circle.copy()
    roi_circle[2] = roi_circle[2] - 10
    
    img_gray_center = img_gray_center.copy()
    imshow_scale(img_gray_center, 0.4, 'img_gray_center')
    im_h, im_w, im_c = bk_bg_grid.shape
    gray_grid = cv.cvtColor(bk_bg_grid, cv.COLOR_BGR2GRAY)
    gray_grid[gray_grid>0] = 255
    
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    imshow_scale(gray_grid, 0.4, 'grid')

    gray_grid = cv.dilate(gray_grid, k) ## 팽창연산
    imshow_scale(gray_grid, 0.4, 'dilate')

    gray_grid = cv.morphologyEx(gray_grid, cv.MORPH_CLOSE, k) ## 닫힘연산
    imshow_scale(gray_grid, 0.4, 'close')
    
    # values projected to X
    p_hori = np.average(gray_grid, axis=0)
    print('sh: ',p_hori.shape) #1920
    # values projected to Y
    p_vert = np.average(gray_grid, axis=1)
    print(roi_circle)
    x_begin = roi_circle[0] - roi_circle[2]
    y_begin = roi_circle[1] - roi_circle[2]
    x_end = roi_circle[0] + roi_circle[2]
    y_end = roi_circle[1] + roi_circle[2]
    
    r = roi_circle[2]
    cx = roi_circle[0]
    cy = roi_circle[1]

    p2_hori = p_hori[x_begin:x_end]
    p2_vert = p_vert[y_begin:y_end]

    dx = np.abs(np.arange(x_begin, x_end, 1, dtype=np.float32) - cx) #원의 중심과 x좌표 거리
    dy = np.abs(np.arange(y_begin, y_end, 1, dtype=np.float32) - cy) #원의 중심과 y좌표 거리

    dx[dx>=r] = r-1
    dy[dy>=r] = r-1

    weight_h = r / np.sqrt(r ** 2 - (dx) ** 2) #가중치값
    weight_v = r / np.sqrt(r ** 2 - (dy) ** 2)
    print(p2_hori.shape[0])
    plt.figure(1, figsize=(5,4))
    plt.subplot(2,1,1)
    plt.plot(range(p2_hori.shape[0]), p2_hori)
    plt.ylim(0, 255)
    plt.subplot(2,1,2)
    plt.plot(range(p2_vert.shape[0]), p2_vert)
    plt.ylim(0, 255)
    plt.tight_layout()
    
    p3_hori = np.multiply(p2_hori, weight_h)
    p3_vert = np.multiply(p2_vert, weight_v)

    plt.figure(2, figsize=(5,4))
    plt.subplot(2, 1, 1)
    plt.plot(range(p3_hori.shape[0]), p3_hori)
    plt.ylim(0, 255)
    plt.subplot(2,1,2)
    plt.plot(range(p3_vert.shape[0]), p3_vert)
    plt.ylim(0, 255)
    plt.tight_layout()
    
    plt.show()
    ###
    peaks_all_h = []
    peaks_all_v = []
    peaks_c4_h = []
    peaks_c4_v = []

    # extract 6 peaks
    # horizontal
    p4_hori = p3_hori
    p4_vert = p3_vert

    # using non-correction version
    p4_hori = p2_hori
    p4_vert = p2_vert

    for i in range(6):
        max_h = np.argmax(p4_hori)
        peaks_all_h.append(max_h) #index (x)
        e1 = max_h - 25
        e2 = max_h + 25
        if e1 < 0:
            e1 = 0
        if e2 >= p4_hori.shape[0]:
            e2 = p4_hori.shape[0]-1
        p4_hori[e1:e2] = 0

        max_v = np.argmax(p4_vert)
        peaks_all_v.append(max_v)
        e1 = max_v - 25
        e2 = max_v + 25
        if e1 < 0:
            e1 = 0
        if e2 >= p4_vert.shape[0]:
            e2 = p4_vert.shape[0]-1
        p4_vert[e1:e2] = 0
    print('peaks_all_h:', peaks_all_h)
    for i in range(4):
        min_dist_x = im_w
        min_dist_val = -1
        for j in range(len(peaks_all_h)): ##6개
            dist = np.abs(peaks_all_h[j]  - int(p4_hori.shape[0]/2)) #원의 중점과의 x값 offset
            if dist < min_dist_x:
                min_dist_val = peaks_all_h[j]
                min_dist_x = dist
        peaks_c4_h.append(min_dist_val)
        peaks_all_h.remove(min_dist_val)

        min_dist_y = im_h
        min_dist_val = -1
        for j in range(len(peaks_all_v)):
            dist = np.abs(peaks_all_v[j]  -  int(p4_vert.shape[0]/2))
            if dist < min_dist_y:
                min_dist_val = peaks_all_v[j]
                min_dist_y = dist
        peaks_c4_v.append(min_dist_val)
        peaks_all_v.remove(min_dist_val)


    arr_h = np.array(peaks_c4_h) + x_begin #중심과 가까운 line x좌표 4개
    arr_v = np.array(peaks_c4_v) + y_begin #중심과 가까운 line y좌표 4개

    ngrid_xmin = int(np.min(arr_h))
    ngrid_xmax = int(np.max(arr_h))
    ngrid_ymin = int(np.min(arr_v))
    ngrid_ymax = int(np.max(arr_v))

    img_mask_grid = gray_grid.copy() * 0
    # img_ori_rgb = cv.cvtColor(img_ori, cv.COLOR_GRAY2BGR)
    # horizontal line
    for i in range(4):
        print((int(arr_h[i]), ngrid_ymin))
        print((int(arr_h[i]), ngrid_ymax))
        cv.line(img_gray_center, (ngrid_xmin, int(arr_v[i])),  (ngrid_xmax, int(arr_v[i])), (255, 255, 255), 3)
        cv.line(img_gray_center, (int(arr_h[i]), ngrid_ymin), (int(arr_h[i]), ngrid_ymax), (255, 255, 255), 3)

        cv.line(img_mask_grid, (ngrid_xmin, int(arr_v[i])), (ngrid_xmax, int(arr_v[i])), (255, 255, 255), 3)
        cv.line(img_mask_grid, (int(arr_h[i]), ngrid_ymin), (int(arr_h[i]), ngrid_ymax), (255, 255, 255), 3)



    mask_supp = cv.bitwise_and(gray_grid.copy(), img_mask_grid) ##detection한 grid와 예측한 grid 일치성
    sum_mask = np.sum(img_mask_grid)
    sum_supp = np.sum(mask_supp)
    ratio_supp = float(sum_supp) / float(sum_mask) * 100.0

    print('sum of grid mask: %d' % sum_mask)
    print('sum of superposition: %d' % sum_supp)
    print('supp ratio: %.2f'%ratio_supp)

    cv.circle(img_gray_center, (int(cx), int(cy)), 10, (255, 0, 0), 3)
    imshow_scale(gray_grid, 0.4, 'result:gray_grid')
    imshow_scale(img_mask_grid, 0.4, 'result:img_mask_grid')
    imshow_scale(img_gray_center, 0.4, 'result:grid_basis')
    imshow_scale(mask_supp, 0.4, 'result:superposition')
    
    return ratio_supp
if __name__ == '__main__':

    sol_name = 'CUB'
    sol_idx = 4
    aug_idx = 0
    img_name = f'./eval/{sol_name}-00{sol_idx}-03-f2-0{aug_idx}.png'
    print(img_name)
    img_rgb = cv.imread(img_name)    
    
    # get dimensions
    im_h, im_w, im_c = img_rgb.shape

    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    clip_size = 450
    img_gray = clip_image_center(img_gray, clip_size)

    # roi rectangle [LT_x, LT_y, RB_x, RB_y]
    roi_rect = np.ones(4) * -1
    roi_circle = roi_detection_circle(img_rgb, img_gray)
    
    if roi_circle[2] != -1:
        roi_rect = roi_rect.astype(np.int)
        roi_rect = np.array([roi_circle[0]-roi_circle[2], roi_circle[1]-roi_circle[2], roi_circle[0]+roi_circle[2], roi_circle[1]+roi_circle[2]])

        img_rgb_rect = cv.rectangle(img_rgb.copy(), (roi_rect[0], roi_rect[1]), (roi_rect[2], roi_rect[3]), (0, 255, 0), 5)

        # masking
        img_rgb_center = masking_circle(img_rgb, roi_circle)
        img_gray_center = masking_circle(img_gray, roi_circle)
        
        imshow_scale(img_rgb_center, 0.4, 'img_center')
        imshow_scale(img_gray_center, 0.4, 'img_center')

    imshow_scale(img_rgb_rect, 0.4, 'rgb')

    ## grid_homogenity calculation
    mmg, msg, smg, ssg = grid_calc(img_gray, roi_rect)
    
    ## segmentation - crystal detection
    particle_sum = circle_segmentation(img_gray_center, roi_circle)
    
    # cross_profile profile
    curve_a, curve_b, curve_c, cross_profile_area, gradient_plot = cross_profile(img_gray_center, roi_circle)
    
 # round 2
    #sol_name = 'COA'
    #sol_idx = 1
    
    img_name =  f'./eval/{sol_name}-00{sol_idx}-02-f2-0{aug_idx}.png'
    #img_name = './%s-%03d-02.jpeg' % (sol_name, sol_idx)
    img_rgb = cv.imread(img_name)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    # Set clip size as amount of solution
    clip_size = 450
    img_gray = clip_image_center(img_gray, clip_size)
    # roi rectangle [LT_x, LT_y, RB_x, RB_y]
    roi_rect = np.ones(4) * -1

    # circle ROI is only available for type3 (white background)
    roi_circle = roi_detection_circle(img_rgb, img_gray)
    # set rectangle roi by circle roi
    if roi_circle[2] != -1:
        roi_rect = roi_rect.astype(np.int8)
        roi_rect = np.array([roi_circle[0]-roi_circle[2], roi_circle[1]-roi_circle[2], roi_circle[0]+roi_circle[2], roi_circle[1]+roi_circle[2]])

        img_rgb_rect = cv.rectangle(img_rgb.copy(), (roi_rect[0], roi_rect[1]), (roi_rect[2], roi_rect[3]), (0, 255, 0), 5)

        # masking
        img_rgb_center = masking_circle(img_rgb, roi_circle)
        img_gray_center = masking_circle(img_gray, roi_circle)

        #imshow_scale(img_rgb_center, 0.4, 'img_center')
        #imshow_scale(c, 0.4, 'img_gray_center')

    #check_detection(img_gray_center.copy(), roi_circle)
    img_grid = line_detection(img_gray_center.copy(), roi_circle)
    bk_bg_grid = black_bg_line(img_gray_center.copy(), roi_circle)

    #grid_profile(img_grid, roi_circle)
    #crystal_detection(img_gray_center, roi_circle)
    #circle_segmentation(img_gray_center, roi_circle)
    supp_ratio = grid_detection(img_gray_center, bk_bg_grid, roi_circle)
    
