import cv2


class SSDclass(object):
    def __init__(self, use_mask):
        self.name = "SSD"
        self.use_mask = use_mask

    @staticmethod
    def calculate_ssd(mat1, mat2):
        return cv2.norm(cv2.absdiff(mat1, mat2), normType=cv2.NORM_L2)

    def run(self, image, template, template_mean, template_std_dev, mask):
        x, y, _ = image.shape
        x_t, y_t, _ = template.shape
        min_value = 1e10
        res_x = -1
        res_y = -1
        for i in range(x - x_t):
            for j in range(y - y_t):
                can = True
                if self.use_mask and mask[i][j] < 128:
                    can = False
                if can:
                    temp = image[i: i + x_t, j: j + y_t, :]
                    value = self.calculate_ssd(temp, template)
                    if value < min_value:
                        res_x = i
                        res_y = j
                        min_value = value
        return res_x, res_y, min_value
