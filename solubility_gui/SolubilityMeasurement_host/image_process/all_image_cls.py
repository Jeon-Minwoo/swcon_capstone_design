import yaml
import os
import csv
import cv2 as cv
import numpy as np


class ImageFeatureExtractor:
    def __init__(self, img: np.array):
        self.img = img
        self.clipped_img_gray = self.clip_image_center(450)

    def clip_image_center(self, clip_size):
        img_gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        gray_clip = np.zeros_like(img_gray)
        h, w = img_gray.shape
        hh = int(h / 2)  # 이미지 중점
        ww = int(w / 2)

        hor_lower = ww - clip_size
        hor_upper = ww + clip_size
        ver_lower = hh - clip_size
        ver_upper = hh + clip_size

        gray_clip[ver_lower:ver_upper, hor_lower:hor_upper] = img_gray[ver_lower:ver_upper, hor_lower:hor_upper]
        return gray_clip

    def roi_detection_circle(self):
        src = self.img.copy()
        src_ = self.img.copy()
        
        clipped_img_gray = self.clipped_img_gray

        gray = cv.medianBlur(clipped_img_gray, 3)
        rows, cols = gray.shape
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.0, rows / 12, None, 20, 20, minRadius=250, maxRadius=350)
        ww = int(cols / 2)
        hh = int(rows / 2)
        min_dist_center = cols * 100 

        roi = np.ones(3) * -1

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv.circle(src_, (i[0], i[1]), i[2], (0, 255, 0), 2)
                dist_center = np.sqrt((i[0] - ww) * (i[0] - ww) + (i[1] - hh) * (i[1] - hh))  # 이미지 중점과 원의 거리
                if dist_center < min_dist_center:
                    min_dist_center = dist_center  # 가장 이미지의 중심과 가까운 원의 중심 찾기
                    if np.abs((i[0] - ww)) < 100 and np.abs((i[1] - hh)) < 100:  # 가장 가까운 원의 중심일때 최소 중심과 차이 100미만
                        roi[0] = i[0]
                        roi[1] = i[1]
                        roi[2] = i[2]

        roi = np.uint16(np.around(roi))
        cv.circle(src_, (roi[0], roi[1]), 3, (0, 0, 255), 3)
        cv.circle(src, (roi[0], roi[1]), roi[2], (255, 255, 255), 5)
        return roi

    def grid_calc(self, roi_rect):
        clipped_img_gray = self.clipped_img_gray.copy()

        grid_size = 80
        cx = (roi_rect[0] + roi_rect[2]) / 2  # 직사각형 좌표를 통해 원의 중점, 반지름 구하기
        cy = (roi_rect[1] + roi_rect[3]) / 2
        radius = (roi_rect[2] - roi_rect[0]) / 2

        list_mean = []
        list_std = []

        for r in range(roi_rect[1], roi_rect[3], grid_size):  # y
            for c in range(roi_rect[0], roi_rect[2], grid_size):  # x
                # 원의 중점과 각 그리드 꼭지점 사이의 거리
                dist_lt = np.sqrt((c - cx) ** 2 + (r - cy) ** 2)
                dist_rt = np.sqrt((c + grid_size - cx) ** 2 + (r - cy) ** 2)
                dist_lb = np.sqrt((c - cx) ** 2 + (r + grid_size - cy) ** 2)
                dist_rb = np.sqrt((c + grid_size - cx) ** 2 + (r + grid_size - cy) ** 2)
                # 각 그리드 꼭지점 까지의 거리 4가지 모두 원의 반지름보다 작은 그리드가 원 안의 그리드..!
                if dist_lt <= radius and dist_rt <= radius and dist_lb <= radius and dist_rb <= radius:
                    patch = clipped_img_gray[r:r + grid_size, c:c + grid_size]  # 그리드 안의 모든 pixel들..
                    mean = np.mean(patch)
                    std = np.std(patch)
                    list_mean = np.append(list_mean, mean)
                    list_std = np.append(list_std, std)
                    #cv.rectangle(gray, (c, r), (c + grid_size, r + grid_size), (255, 255, 255), 3)

        #cv.circle(gray, (int(cx), int(cy)), int(radius), (0, 0, 0), 3)
        mmg = np.mean(np.array(list_mean))
        msg = np.mean(np.array(list_std))
        smg = np.std(np.array(list_mean))
        ssg = np.std(np.array(list_std))

        #print('mean of mean of grid: %f' % mmg)
        #print('mean of std of grid: %f' % msg)
        #print('std of mean of grid: %f' % smg)
        #print('std of std of grid: %f' % ssg)
        return mmg, msg, smg, ssg

    def circle_segmentation(self, roi_circle):
        img_c = self.clipped_img_gray
        img_c = ImageFeatureExtractor.masking_circle(img_c, roi_circle)

        img_c = img_c.copy()
        im_rgb_c = cv.cvtColor(img_c, cv.COLOR_GRAY2BGR)
        im_c = cv.medianBlur(img_c, 5)
        im_c = ImageFeatureExtractor.saturate_contrast(im_c, 1.0)
        im_bin = cv.adaptiveThreshold(im_c, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 151, 15)
        im_bin = ImageFeatureExtractor.masking_circle(im_bin, roi_circle)

        # 원을 둘러싼 직사각형 범위
        for r in range(roi_circle[1] - roi_circle[2], roi_circle[1] + roi_circle[2]):
            for c in range(roi_circle[0] - roi_circle[2], roi_circle[0] + roi_circle[2]):
                if im_bin[r, c] > 100:
                    cv.circle(im_rgb_c, (c, r), 3, (0, 255, 255), -1)

        particle_sum = np.sum(im_bin / 255)  #이진 threshold라 0 or 255의 값을 가진다
        print('particle sum: %d' % particle_sum)
        return particle_sum

    def cross_profile(self, roi_circle):
        img_c = self.clipped_img_gray
        img_c = ImageFeatureExtractor.masking_circle(img_c, roi_circle)

        img_c = img_c.copy()
        img_c = cv.medianBlur(img_c, 5)
        img_c_rgb = cv.cvtColor(img_c, cv.COLOR_GRAY2BGR)

        # move circle center to image center
        h, w = img_c.shape
        cx = int(w / 2)
        cy = int(h / 2)
        offset_x = cx - roi_circle[0]
        offset_y = cy - roi_circle[1]

        cv.circle(img_c_rgb, (cx, cy), 5, (255, 0, 0), 3)  # blue - 이미지의 중점
        # red - 원의 중점 // calibration이 잘 되지 않으면 차이가 발생할 수 밖에없다
        cv.circle(img_c_rgb, (roi_circle[0], roi_circle[1]), 5, (0, 0, 255), 3)
        # imshow_scale(img_rgb, 0.4, 'cross')

        M = np.float64([[1, 0, offset_x], [0, 1, offset_y]])
        # 2차원 이동 matrix 변환 - 원의 중점과 이미지의 중점 calibration
        img_c_trans = cv.warpAffine(img_c, M, (w, h))
        img_c_trans_ = img_c_trans.copy()
        cv.circle(img_c_trans_, (cx, cy), 5, (255, 0, 0), 3)

        cross_profile = np.array([])

        for i in range(0, 360, 30):
            R = cv.getRotationMatrix2D((cx, cy), i, 1)
            img_gray_rot = cv.warpAffine(img_c_trans, R, (w, h))
            profile = img_gray_rot[cy, cx - roi_circle[2] + 20:cx + roi_circle[2] - 20]
            profile = np.reshape(profile, (1, -1))
            cross_profile = np.vstack([cross_profile, profile]) if cross_profile.size else profile

        cross_profile_mean = np.mean(cross_profile, axis=0)
        cross_profile_std = np.std(cross_profile, axis=0)
        cross_profile_area = 2 * np.sum(cross_profile_std)

        curve_x = range(cross_profile.shape[1])
        curve_y = cross_profile_mean
        curve_fitted = np.polyfit(curve_x, curve_y, 2)
        curve_a, curve_b, curve_c = curve_fitted  # ax^2 + bx + c = y
        print('curve coefficient: ', curve_fitted)
        curve_fit_y = curve_a * curve_x * curve_x + curve_b * curve_x + curve_c

        min_plot_x = np.argmin(curve_fit_y).astype(np.int64)
        min_plot_y = cross_profile_mean[min_plot_x]
        max_plot_x = 0
        max_plot_y = curve_c.astype(np.int64)

        gradient_plot = np.round(np.abs((max_plot_y - min_plot_y) / (max_plot_x - min_plot_x)), 4)
        print(max_plot_x, max_plot_y, min_plot_x, min_plot_y, gradient_plot)
        print('std range:', cross_profile_area)

        return curve_a, curve_b, curve_c, cross_profile_area, gradient_plot

    def black_bg_line(self, roi_circle):
        roi_circle[2] = roi_circle[2] - 10
    
        img_c = self.clipped_img_gray
        img_c = ImageFeatureExtractor.masking_circle(img_c, roi_circle)

        img_c_rgb = cv.cvtColor(img_c, cv.COLOR_GRAY2BGR)
        bk_bg_line = np.zeros_like(img_c_rgb)
        img_c = ImageFeatureExtractor.saturate_contrast(img_c, 1.5)
        img_c = cv.GaussianBlur(img_c, (3, 3), 0)

        th = cv.adaptiveThreshold(img_c, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 101, 25)
        th = ImageFeatureExtractor.masking_circle(th, roi_circle)
        edges = cv.Canny(th, 0, 100)

        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(bk_bg_line, (x1, y1), (x2, y2), (0, 255, 255), 2)

        return bk_bg_line

    def grid_detection(self, bk_bg_grid, roi_circle):
        roi_circle[2] = roi_circle[2] - 10

        img_c = self.clipped_img_gray
        img_c = ImageFeatureExtractor.masking_circle(img_c, roi_circle)

        im_h, im_w, im_c = bk_bg_grid.shape
        gray_grid = cv.cvtColor(bk_bg_grid, cv.COLOR_BGR2GRAY)
        gray_grid[gray_grid > 0] = 255

        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        gray_grid = cv.dilate(gray_grid, k)  # 팽창연산
        gray_grid = cv.morphologyEx(gray_grid, cv.MORPH_CLOSE, k)  # 닫힘연산

        # values projected to X
        p_hori = np.average(gray_grid, axis=0)
        # values projected to Y
        p_vert = np.average(gray_grid, axis=1)

        x_begin = roi_circle[0] - roi_circle[2]
        y_begin = roi_circle[1] - roi_circle[2]
        x_end = roi_circle[0] + roi_circle[2]
        y_end = roi_circle[1] + roi_circle[2]

        r = roi_circle[2]
        cx = roi_circle[0]
        cy = roi_circle[1]

        p2_hori = p_hori[x_begin:x_end]
        p2_vert = p_vert[y_begin:y_end]

        dx = np.abs(np.arange(x_begin, x_end, 1, dtype=np.float64) - cx)  # 원의 중심과 x좌표 거리
        dy = np.abs(np.arange(y_begin, y_end, 1, dtype=np.float64) - cy)  # 원의 중심과 y좌표 거리

        dx[dx >= r] = r - 1
        dy[dy >= r] = r - 1

        peaks_all_h = []
        peaks_all_v = []
        peaks_c4_h = []
        peaks_c4_v = []

        # using non-correction version
        p4_hori = p2_hori
        p4_vert = p2_vert

        for i in range(6):
            max_h = np.argmax(p4_hori)
            peaks_all_h.append(max_h)  # index (x)
            e1 = max_h - 25
            e2 = max_h + 25
            if e1 < 0:
                e1 = 0
            if e2 >= p4_hori.shape[0]:
                e2 = p4_hori.shape[0] - 1
            p4_hori[e1:e2] = 0

            max_v = np.argmax(p4_vert)
            peaks_all_v.append(max_v)
            e1 = max_v - 25
            e2 = max_v + 25
            if e1 < 0:
                e1 = 0
            if e2 >= p4_vert.shape[0]:
                e2 = p4_vert.shape[0] - 1
            p4_vert[e1:e2] = 0
        print('peaks_all_h:', peaks_all_h)
        for i in range(4):
            min_dist_x = im_w
            min_dist_val = -1
            for j in range(len(peaks_all_h)):  # 6개
                dist = np.abs(peaks_all_h[j] - int(p4_hori.shape[0] / 2))  # 원의 중점과의 x값 offset
                if dist < min_dist_x:
                    min_dist_val = peaks_all_h[j]
                    min_dist_x = dist
            peaks_c4_h.append(min_dist_val)
            peaks_all_h.remove(min_dist_val)

            min_dist_y = im_h
            min_dist_val = -1
            for j in range(len(peaks_all_v)):
                dist = np.abs(peaks_all_v[j] - int(p4_vert.shape[0] / 2))
                if dist < min_dist_y:
                    min_dist_val = peaks_all_v[j]
                    min_dist_y = dist
            peaks_c4_v.append(min_dist_val)
            peaks_all_v.remove(min_dist_val)

        arr_h = np.array(peaks_c4_h) + x_begin  # 중심과 가까운 line x좌표 4개
        arr_v = np.array(peaks_c4_v) + y_begin  # 중심과 가까운 line y좌표 4개

        ngrid_xmin = int(np.min(arr_h))
        ngrid_xmax = int(np.max(arr_h))
        ngrid_ymin = int(np.min(arr_v))
        ngrid_ymax = int(np.max(arr_v))

        img_mask_grid = gray_grid.copy() * 0
        # horizontal line
        for i in range(4):
            print((int(arr_h[i]), ngrid_ymin))
            print((int(arr_h[i]), ngrid_ymax))
            cv.line(img_c, (ngrid_xmin, int(arr_v[i])), (ngrid_xmax, int(arr_v[i])), (255, 255, 255), 3)
            cv.line(img_c, (int(arr_h[i]), ngrid_ymin), (int(arr_h[i]), ngrid_ymax), (255, 255, 255), 3)

            cv.line(img_mask_grid, (ngrid_xmin, int(arr_v[i])), (ngrid_xmax, int(arr_v[i])), (255, 255, 255), 3)
            cv.line(img_mask_grid, (int(arr_h[i]), ngrid_ymin), (int(arr_h[i]), ngrid_ymax), (255, 255, 255), 3)

        mask_supp = cv.bitwise_and(gray_grid.copy(), img_mask_grid)  # detection한 grid와 예측한 grid 일치성
        sum_mask = np.sum(img_mask_grid)
        sum_supp = np.sum(mask_supp)
        ratio_supp = float(sum_supp) / float(sum_mask) * 100.0

        print('sum of grid mask: %d' % sum_mask)
        print('sum of superposition: %d' % sum_supp)
        print('supp ratio: %.2f' % ratio_supp)

        cv.circle(img_c, (int(cx), int(cy)), 10, (255, 0, 0), 3)
        return ratio_supp

    def processing_whitebg(self):
        img_c = self.clipped_img_gray
        
        # roi rectangle [LT_x, LT_y, RB_x, RB_y]
        roi_rect = np.ones(4) * -1
        roi_circle = self.roi_detection_circle()

        if roi_circle[2] != -1:
            roi_rect = np.array(
                [roi_circle[0] - roi_circle[2], roi_circle[1] - roi_circle[2], roi_circle[0] + roi_circle[2],
                 roi_circle[1] + roi_circle[2]])

            img_gray_center = ImageFeatureExtractor.masking_circle(img_c, roi_circle)

        # grid_homogenity calculation
        mmg, msg, smg, ssg = self.grid_calc(roi_rect)

        # segmentation - crystal detection
        particle_sum = self.circle_segmentation(roi_circle)

        # cross_profile profile
        curve_a, curve_b, curve_c, cross_profile_area, gradient_plot = self.cross_profile(roi_circle)

        return [[mmg, msg, smg, ssg],
                particle_sum,
                [curve_a, curve_b, curve_c, cross_profile_area, gradient_plot]]

    def processing_checkpat(self):
        img_c = self.clipped_img_gray

        # circle ROI is only available for type3 (white background)
        roi_circle = self.roi_detection_circle()
        # set rectangle roi by circle roi
        if roi_circle[2] != -1:
            img_gray_center = ImageFeatureExtractor.masking_circle(img_c, roi_circle)

        bk_bg_grid = self.black_bg_line(roi_circle)
        supp_ratio = self.grid_detection(bk_bg_grid, roi_circle)

        return supp_ratio

    def all_images_run(self):
        path = './SolubilityMeasurement_host/image_process/'
        image_list = [img for img in os.listdir(path + 'eval') if '.jpeg' in img]
        image_list = sorted(image_list)
        print(image_list)
        for i in image_list:
            path = f'./eval/{i}'
            if i[8:10] == '01':
                continue
            elif i[8:10] == '03':
                print("processing whitebg -- ", i)
                result1 = self.processing_whitebg()
                mmg, msg, smg, ssg = result1[0]
                curve_a, curve_b, curve_c, cross_profile_area, gradient_plot = result1[2]
                particle_sum = result1[1]

                solution_info = dict(
                    Grid_homogeneity=dict(
                        mmg=float(mmg),
                        msg=float(msg),
                        smg=float(smg),
                        ssg=float(ssg)
                    ),
                    Profile=dict(
                        curve_a=float(curve_a),
                        curve_b=float(curve_b),
                        curve_c=float(curve_c),
                        area=float(cross_profile_area),
                        gradient=float(gradient_plot)
                    ),
                    Segmentation=dict(
                        particle_sum=int(particle_sum)
                    )
                )
                with open(path + f'solution_info/{i[:7]}.yml', 'a') as outfile:
                    yaml.dump(solution_info, outfile, default_flow_style=False)

            elif i[8:10] == '02':
                print("processing checkptn --", i)
                result2 = self.processing_checkpat(path)
                supp_ratio = result2

                solution_info = dict(
                    Name=i[:7],
                    check_detection=dict(
                        supp_ratio=supp_ratio
                    )
                )

                with open(path + f'./solution_info/{i[:7]}.yml', 'w') as outfile:
                    yaml.dump(solution_info, outfile, default_flow_style=False)

    @staticmethod    
    def saturate_contrast(img, num):
        pic = img.copy()
        pic = pic.astype('int32')
        pic = np.clip(pic + (pic - 128) * num, 0, 255)
        pic = pic.astype('uint8')
        return pic
    
    @staticmethod
    def masking_circle(img, roi_circle):
        img_mask = np.zeros_like(img)
        img_mask = cv.circle(img_mask, (roi_circle[0], roi_circle[1]), roi_circle[2], (255, 255, 255), -1)
        # bitwise_and 연산을 통한 마스킹
        img_center = cv.bitwise_and(img.copy(), img_mask)
        return img_center

    @staticmethod
    def yaml2csv():
        path = './SolubilityMeasurement_host/image_process/'
        info_file_name = os.listdir(path + './solution_info')
        info_file_name = sorted(info_file_name)
        print(info_file_name)
        cnt = 0
        for f in info_file_name:
            with open(path + f'./solution_info/{f}') as yml_info:
                info = yaml.safe_load(yml_info)
                info_row = [info['Name'], info['check_detection']['supp_ratio'], info['Grid_homogeneity']['mmg'],
                            info['Grid_homogeneity']['msg'],
                            info['Grid_homogeneity']['smg'], info['Grid_homogeneity']['ssg'], info['Profile']['area'],
                            info['Profile']['curve_a'],
                            info['Profile']['curve_b'], info['Profile']['curve_c'], info['Profile']['gradient'],
                            info['Segmentation']['particle_sum']]
            if cnt == 0:
                with open(path + "./solution_data.csv", 'w') as data:
                    wr = csv.writer(data, lineterminator='\n')
                    wr.writerow(
                        ['Name', 'supp_ratio', 'mmg', 'msg', 'smg', 'ssg', 'area', 'curve_a', 'curve_b', 'curve_c',
                         'gradient', 'particle_sum'])
                    wr.writerow(info_row)
            else:
                with open(path + "./solution_data.csv", 'a') as data:
                    wr = csv.writer(data, lineterminator='\n')
                    wr.writerow(info_row)
            cnt += 1

    # For test
    @staticmethod
    def imshow_scale(img, s, title):
        img_scale = cv.resize(img, dsize=(0, 0), fx=s, fy=s, interpolation=cv.INTER_LINEAR)
        cv.imshow(title, img_scale)
        cv.waitKey(0)
        cv.destroyAllWindows()