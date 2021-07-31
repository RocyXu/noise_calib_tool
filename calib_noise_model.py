# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        
# Author: xurongxin
# Version:          
# Created:          
# Description:  Used to get ROI region
# Function List:    
# History:
#       <author>        <version>       <time>      <desc>
# ------------------------------------------------------------------
import glob
import cv2
from matplotlib import scale
import numpy as np
from numpy.core.fromnumeric import mean, size
from sklearn import linear_model
import matplotlib.pyplot as plt

_WINDOW_1 = "1"
WINDOW_DOWN_TIMES = 4
raw=None
point_lt=None
point_rb=None
rects = []
neighbour_size = 4

def on_mouse(event, x, y, flags, param):
    global point_lt, w, h, raw
    raw2 = raw.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        point_lt = [x, y]
        cv2.circle(raw2, point_lt, 10, (0, 255, 0), 5)
        cv2.imshow(_WINDOW_1, raw2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.rectangle(raw2, point_lt, (x, y), (255, 0, 0), thickness=5)
        cv2.imshow(_WINDOW_1, raw2)
    elif event == cv2.EVENT_LBUTTONUP:
        point_rb = [x, y]
        cv2.rectangle(raw2, point_lt, point_rb, (0, 0, 255), thickness=5)
        cv2.imshow(_WINDOW_1, raw2)
        if abs(point_rb[0] - point_lt[0]) > 3 and abs(point_rb[1] - point_lt[1]) > 3:
            rects.append([point_lt, point_rb])
        raw = raw2


def subpixel(in_data):
    cat = np.stack((
        in_data[0::2, 0::2],
        in_data[0::2, 1::2],
        in_data[1::2, 0::2],
        in_data[1::2, 1::2]), axis=2)

    return cat


def reshape_for_spatial_stat(in_data, neighbour_size):
    in_data_chw = np.transpose(in_data, (2, 0, 1)) # c h w
    c, h, w = in_data_chw.shape
    out_list = []
    for i in range(neighbour_size * neighbour_size):
        x = i % neighbour_size
        y = i // neighbour_size

        crop = in_data_chw[:, y:h//neighbour_size*neighbour_size:neighbour_size, x:w//neighbour_size*neighbour_size:neighbour_size]
        out_list.append(crop)

    in_data_chwn = np.stack(out_list, axis=3)
    
    return in_data_chwn


def calib_noise_model(frms, h, w, rects):
    ### memory limited, recursively calc mean, var
    mean = 0
    var = 0
    for i, frm in enumerate(frms):
        print("\033[1;36;40m[INFO] \033[0m", "processing {}".format(i))
        raw = np.fromfile(frm, np.uint16).reshape(int(h), int(w)).astype(np.float32)

        if i == 0:
            mean_pre = mean
        else:
            mean_pre = mean.copy()
        mean = mean_pre + (raw - mean_pre) / (i + 1)#(1 - 1 / (i + 1)) * mean_pre + 1 / (i + 1) * raw
        var = i / ((i + 1) ** 2) * np.power(raw - mean_pre, 2) + i / (i + 1) * var#var + (raw - mean_pre) * (raw - mean)

    mean_sub = subpixel(mean)
    var_sub = subpixel(var)

    crop_mean_sub_list = []
    crop_var_sub_list = []
    for rect in rects:
        lt_xy = rect[0]
        rb_xy = rect[1]

        lt_x = lt_xy[0] // 2 # coordinate in sub raw
        lt_y = lt_xy[1] // 2
        rb_x = rb_xy[0] // 2 # coordinate in sub raw
        rb_y = rb_xy[1] // 2

        crop_mean_sub = mean_sub[lt_y: rb_y, lt_x: rb_x, :]#.reshape(-1, 4)# h, w, c
        crop_var_sub = var_sub[lt_y: rb_y, lt_x: rb_x, :]#.reshape(-1, 4)# h, w, c

        crop_mean_sub = reshape_for_spatial_stat(crop_mean_sub, neighbour_size)
        crop_var_sub = reshape_for_spatial_stat(crop_var_sub, neighbour_size)
        
        crop_mean_sub = (1 / neighbour_size / neighbour_size) * crop_mean_sub
        crop_var_sub = (1 / neighbour_size / neighbour_size) * crop_var_sub

        crop_mean_sub = np.sum(crop_mean_sub, axis=3).reshape((4, -1))
        crop_var_sub = np.sum(crop_var_sub, axis=3).reshape((4, -1))
        crop_mean_sub = np.transpose(crop_mean_sub, (1, 0))
        crop_var_sub = np.transpose(crop_var_sub, (1, 0))

        crop_mean_sub_list.append(crop_mean_sub)
        crop_var_sub_list.append(crop_var_sub)

    rect_means = np.concatenate(crop_mean_sub_list, axis=0)    
    rect_vars = np.concatenate(crop_var_sub_list, axis=0)    

    return rect_means, rect_vars


def calc_linear_model(means, vars):
    ransac = linear_model.RANSACRegressor(
        max_trials=1000)
    ransac.fit(means, vars)
    
    return ransac, ransac.estimator_.coef_, ransac.estimator_.intercept_


def plot(means, vars, ransac, k, b):
    x = np.arange(1, 500).reshape(-1, 1)
    y = ransac.predict(x)
    plt.figure()
    plt.subplot(2,2,1)
    plt.scatter(means[:, 0:1], vars[:, 0:1], s=5)
    plt.plot(x, y)
    plt.subplot(2,2,2)
    plt.scatter(means[:, 1:2], vars[:, 1:2], s=5)
    plt.plot(x, y)
    plt.subplot(2,2,3)
    plt.scatter(means[:, 2:3], vars[:, 2:3], s=5)
    plt.plot(x, y)
    plt.subplot(2,2,4)
    plt.scatter(means[:, 3:], vars[:, 3:], s=5)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    raw_files1 = glob.glob(r"F:\1-Data\GN1_Noise_model_calib\20210103_063838_IFE_4080x3060_unpackedRAW/*.raw")
    #raw_files1 = glob.glob(r"F:/1-Data/GN1_Noise_model_calib/20210102_104728_IFE_4080x3060_unpackedRA/*.raw")

    w = 4080#input("Raw width:")
    w = int(w)
    h = 3060#input("Raw height:")
    h = int(h)
    bits = 10#input("Bits:")
    bits = int(bits)

    raw = np.fromfile(raw_files1[0], np.uint16).reshape(int(h), int(w)) / ((1 << (bits - 8)) - 1)
    raw = cv2.cvtColor(raw.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    orig_raw = raw.copy()
    
    cv2.namedWindow(_WINDOW_1, cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(_WINDOW_1, w // WINDOW_DOWN_TIMES, h // WINDOW_DOWN_TIMES)
    print("\033[1;36;40m[INFO] \033[0m", "Start your performance, draw rectangles where to calc mean, var temporally!")
    while True:
        cv2.setMouseCallback(_WINDOW_1, on_mouse)
        cv2.imshow(_WINDOW_1, raw)
        key = cv2.waitKey(0)
        if key == 13 or key == 32:# press space and enter to quit
            break
        elif key == ord('r'):
            raw = orig_raw
            rects = []
    print(rects)
    #rects = [[[1180, 776], [1348, 920]], [[1468, 784], [1640, 916]], [[1804, 796], [1948, 920]], [[2100, 800], [2256, 928]], [[2392, 808], [2544, 932]], [[2680, 800], [2836, 916]], [[1184, 1104], [1328, 1236]], [[1188, 1404], [1332, 1528]], [[1188, 1696], [1348, 1848]], [[1492, 1112], [1656, 1248]], [[1508, 1400], [1640, 1544]], [[1512, 1708], [1648, 1844]], [[1788, 1120], [1948, 1232]], [[1792, 1404], [1932, 1536]], [[1804, 1692], [1956, 1828]], [[2108, 1112], [2236, 1212]], [[2424, 1116], [2552, 1220]], [[2696, 1096], [2848, 1228]], [[2704, 1392], [2832, 1508]], [[2420, 1400], [2556, 1516]], [[2112, 1408], [2232, 1516]], [[2116, 1688], [2240, 1816]], [[2416, 1692], [2536, 1820]], [[2708, 1696], [2824, 1816]]]

    means1, vars1 = calib_noise_model(raw_files1, h, w, rects)
    #means2, vars2 = calib_noise_model(raw_files2, h, w, rects)

    ransac, k, b = calc_linear_model(means1[:, 0:1], vars1[:, 0:1])
    plot(means1, vars1, ransac, k, b)
    print(k, b)