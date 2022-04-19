import cv2
import numpy as np


class NCCclass(object):
    def __init__(self, use_mask):
        self.name = "NCC"
        self.use_mask = use_mask

    @staticmethod
    def calculate_NCC(mat1, mat2, mat2_mean, mat2_std_dev):
        mat1_mean, mat1_std_dev = cv2.meanStdDev(mat1)
        mat1_temp = np.ones(mat1.shape)
        mat2_temp = np.ones(mat2.shape)
        mul = (mat2_std_dev * mat1_std_dev)
        for i in range(3):
            mat1_temp[:, :, i] = mat1[:, :, i] - mat1_mean[i][0]
            mat2_temp[:, :, i] = mat2[:, :, i] - mat2_mean[i][0]
        res = np.multiply(mat1_temp, mat2_temp)
        for i in range(3):
            res[:, :, i] = res[:, :, i] / mul[i][0]
        res = res.sum()
        return res

    def run(self, image, template, template_mean, template_std_dev, mask):
        x, y, _ = image.shape
        x_t, y_t, _ = template.shape
        max_value = -1e10
        res_x = -1
        res_y = -1
        for i in range(x - x_t):
            for j in range(y - y_t):
                can = True
                if self.use_mask and mask[i][j] < 128:
                    can = False
                if can:
                    temp = image[i: i + x_t, j: j + y_t, :]
                    value = self.calculate_NCC(temp, template, template_mean, template_std_dev)
                    if value > max_value:
                        res_x = i
                        res_y = j
                        max_value = value
        return res_x, res_y, max_value
