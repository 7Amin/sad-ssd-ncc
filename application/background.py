import cv2
import numpy as np


def find_movement(img, background):
    th = 80
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((21, 21), np.uint8)
    _, mask_movement = cv2.threshold(cv2.absdiff(img_gray, background_gray), th, 255, cv2.THRESH_BINARY)
    mask_movement = cv2.dilate(mask_movement, kernel, iterations=5)

    # cv2.imshow("mask", cv2.resize(mask_movement, (960, 540)))
    # cv2.imshow("original", cv2.resize(img_gray, (960, 540)))
    # cv2.imshow("background", cv2.resize(background_gray, (960, 540)))
    # cv2.waitKey()
    return mask_movement

