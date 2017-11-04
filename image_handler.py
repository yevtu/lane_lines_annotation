# coding=utf-8
import math
import cv2
import numpy as np


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size=3):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    ignore_mask_color = (255,) * img.shape[2] if len(img.shape) > 2 else 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_line(img, k, b, color, thickness):
    '''
    :param k: line's slope
    :param b: line's bias
    '''
    if np.isnan(b) or np.isnan(k):
        return

    y1 = img.shape[0]
    x1 = int((y1 - b) // k)
    y2 = int(y1 / 1.65)
    x2 = int((y2 - b) // k)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def get_lines_params(hough_lines, x_center, slope_threshold=0.5):
    left_lines_k, left_lines_b, right_lines_k, right_lines_b = [], [], [], []

    if not hough_lines is None:
        for line in hough_lines:
            for x1, y1, x2, y2 in line:
                is_left_line = x_center > x1 and x_center > x2
                is_right_line = x_center < x1 and x_center < x2

                if is_left_line or is_right_line:
                    slope = (y2 - y1) / (x2 - x1)
                    if math.isinf(slope) or abs(slope) < slope_threshold:
                        continue
                    bias = y2 - slope * x2
                    if slope < 0 and is_left_line:
                        left_lines_k.append(slope)
                        left_lines_b.append(bias)
                    if slope > 0 and is_right_line:
                        right_lines_k.append(slope)
                        right_lines_b.append(bias)

    return {
        'lk': np.mean(left_lines_k),
        'lb': np.mean(left_lines_b),
        'rk': np.mean(right_lines_k),
        'rb': np.mean(right_lines_b)
    }

def hough_lines(img, prev_lines_params, rho, theta, threshold, min_line_len, max_line_gap, slope_threshold):
    """
    :param img: output of a Canny transform
    :return: an image with hough lines drawn
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    lines_params = get_lines_params(lines, img.shape[1] // 2, slope_threshold)
    mean_params = lines_params.copy()

    if not prev_lines_params is None:
        for param, value in mean_params.items():
            mean_params[param] = np.nanmean([value, prev_lines_params[param]])

    draw_line(line_img, mean_params['lk'], mean_params['lb'], color=[0, 255, 0], thickness=10)
    draw_line(line_img, mean_params['rk'], mean_params['rb'], color=[0, 255, 0], thickness=10)

    return line_img, lines_params

# Python 3 has support for cool math symbols :)
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    NOTE: initial_img and img must be the same shape!
    :param img: is the output of the hough_lines() - black image with lines drawn on it
    :param initial_img: should be the image before any processing
    :return: initial_img * α + img * β + λ
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)