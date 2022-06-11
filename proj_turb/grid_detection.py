import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def imshow_scale(img, s, title):
    img_scale = cv.resize(img, dsize=(0, 0), fx=s, fy=s, interpolation=cv.INTER_LINEAR)
    cv.imshow(title, img_scale)
    cv.waitKey(0)

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
    print(roi)
    
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
    print(roi)
    cv.circle(src, (roi[0], roi[1]), roi[2], (255, 255, 255), 5)
    imshow_scale(src_, 0.4, 'all circle')
    imshow_scale(src, 0.4, 'circle roi')
    
    return roi

if __name__ == '__main__':

    # sol_name = 'POA'
    # sol_idx = 2
    #
    sol_name = 'POA'
    sol_idx = 4

    grid_name = './grid_img/bk_bg/%s-%03d_1.png'%(sol_name, sol_idx)
    img_grid = cv.imread(grid_name)
    im_h, im_w, im_c = img_grid.shape
    ww = int(im_w/2)
    hh = int(im_h/2)

    ori_name = './grid_img/orgin/%s-%03d_0.png'%(sol_name, sol_idx)
    img_ori = cv.imread(ori_name)

    gray_grid = cv.cvtColor(img_grid, cv.COLOR_BGR2GRAY)
    gray_grid[gray_grid>0] = 255

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)) ## 구조화 요소 커널 (타원형)

    imshow_scale(gray_grid, 0.4, 'grid')

    gray_grid = cv.dilate(gray_grid, k) ## 팽창연산

    imshow_scale(gray_grid, 0.4, 'dilate')

    gray_grid = cv.morphologyEx(gray_grid, cv.MORPH_CLOSE, k) ## 닫힘연산

    imshow_scale(gray_grid, 0.4, 'open')

    # generate vertical/horizontal profile
    y_begin = 0
    y_end = im_h
    x_begin = 0
    x_end = im_w

    # values projected to X
    p_hori = np.average(gray_grid, axis=0)
    print('sh: ',p_hori.shape) #1920
    # values projected to Y
    p_vert = np.average(gray_grid, axis=1)

    # MUST be revised!!!
    # using roi_circle!!
    for i in range(im_w):
        if p_hori[i] > 0:
            x_begin = i
            break

    for i in range(im_w-1, 0, -1):
        if p_hori[i] > 0:
            x_end = i
            break

    for i in range(im_h):
        if p_vert[i] > 0:
            y_begin = i
            break

    for i in range(im_h - 1, 0, -1):
        if p_vert[i] > 0:
            y_end = i
            break

    print(x_begin)
    print(x_end)
    print(y_begin)
    print(y_end)

    r = 0.25 * (y_end - y_begin) + 0.25 * (x_end - x_begin)
    cx = 0.5 * (x_end + x_begin)
    cy = 0.5 * (y_end + y_begin)

    p2_hori = p_hori[x_begin:x_end]
    p2_vert = p_vert[y_begin:y_end]

    dx = np.abs(np.arange(x_begin, x_end, 1, dtype=np.float) - cx) #원의 중심과 x좌표 거리
    dy = np.abs(np.arange(y_begin, y_end, 1, dtype=np.float) - cy) #원의 중심과 y좌표 거리

    dx[dx>=r] = r-1
    dy[dy>=r] = r-1

    weight_h = r / np.sqrt(r ** 2 - (dx) ** 2) #가중치값
    weight_v = r / np.sqrt(r ** 2 - (dy) ** 2)

    print(r)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(range(p2_hori.shape[0]), p2_hori)
    plt.ylim(0, 255)
    plt.subplot(1,2,2)
    plt.plot(range(p2_vert.shape[0]), p2_vert)
    plt.ylim(0, 255)

    p3_hori = np.multiply(p2_hori, weight_h)
    p3_vert = np.multiply(p2_vert, weight_v)

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.plot(range(p3_hori.shape[0]), p3_hori)
    plt.ylim(0, 255)
    plt.subplot(1,2,2)
    plt.plot(range(p3_vert.shape[0]), p3_vert)
    plt.ylim(0, 255)

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
            dist = np.abs(peaks_all_h[j]  - int(p4_hori.shape[0]/2)) #이미지 중점과의 x값 offset
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
        cv.line(img_ori, (ngrid_xmin, int(arr_v[i])),  (ngrid_xmax, int(arr_v[i])), (255, 255, 255), 3)
        cv.line(img_ori, (int(arr_h[i]), ngrid_ymin), (int(arr_h[i]), ngrid_ymax), (255, 255, 255), 3)

        cv.line(img_mask_grid, (ngrid_xmin, int(arr_v[i])), (ngrid_xmax, int(arr_v[i])), (255, 255, 255), 3)
        cv.line(img_mask_grid, (int(arr_h[i]), ngrid_ymin), (int(arr_h[i]), ngrid_ymax), (255, 255, 255), 3)



    mask_supp = cv.bitwise_and(gray_grid.copy(), img_mask_grid) ##detection한 grid와 예측한 grid 일치성
    sum_mask = np.sum(img_mask_grid)
    sum_supp = np.sum(mask_supp)
    ratio_supp = float(sum_supp) / float(sum_mask) * 100.0

    print('sum of grid mask: %d' % sum_mask)
    print('sum of superposition: %d' % sum_supp)
    print('supp ratio: %.2f'%ratio_supp)

    cv.circle(img_ori, (int(cx), int(cy)), 10, (255, 0, 0), 3)
    imshow_scale(gray_grid, 0.4, 'result:gray_grid')
    imshow_scale(img_ori, 0.4, 'result:grid_basis')
    imshow_scale(mask_supp, 0.4, 'result:superposition')


    # align 4

    # make new grid

    # masking
