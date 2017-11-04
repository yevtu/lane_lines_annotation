import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from collections import deque
from image_handler import *

class Annotator:
    def __init__(self, in_path, out_path):
        self._frame_n = 0
        self._in_path = in_path
        self._out_path = out_path
        self._prev_lines_params = deque([], 10)
    
    def process_image(self, image):
        result = self.detect_lane_lines(image)
        self._frame_n += 1
        return result

    def process_video(self):
        clip = VideoFileClip(self._in_path)
        result = clip.fl_image(self.process_image)
        result.write_videofile(self._out_path, audio=False)

    def get_mean_params(self):
        if not len(self._prev_lines_params):
            return None

        params = pd.DataFrame(list(self._prev_lines_params))
        means = {}
        for col in params.columns:
            means[col] = np.nanmean(params[col].values)
        return means

    def detect_lane_lines(self, image):
        x_size, y_size = image.shape[1], image.shape[0]

        blur_gray = gaussian_blur(grayscale(image), kernel_size=3)
        canny_image = canny(blur_gray, low_threshold=10, high_threshold=200)

        left_bottom = (int(0.2 * x_size), int(0.9 * y_size))
        left_upper = (x_size // 2 - 25, int(0.6 * y_size))
        right_upper = (x_size // 2 + 25, int(0.6 * y_size))
        right_bottom = (x_size, int(0.9 * y_size))
        vertices = np.array([[left_bottom, left_upper, right_upper, right_bottom]], dtype=np.int32)
        masked_image = region_of_interest(canny_image, vertices)

        if not self._frame_n % 2:
            self._lines, params = hough_lines(masked_image,
                                        self.get_mean_params(),
                                        rho=1,
                                        theta=np.pi / 180,
                                        threshold=30,
                                        min_line_len=100,
                                        max_line_gap=120,
                                        slope_threshold=0.55)
            self._prev_lines_params.append(params)

        result = weighted_img(self._lines, image)
        return result