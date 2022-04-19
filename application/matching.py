import cv2
import argparse
from factory import create
from background import find_movement


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-pn", '--part_number', type=int, default=2, help='part number')
parser.add_argument("-in", "--image_number", type=int, default=420, help="image number")
parser.add_argument("-a", "--algorithm", type=str, default='NCC', help="algorithm")
parser.add_argument("-um", "--use_mask", type=int, default=0, help="use mask")
args = parser.parse_args()

part_number = args.part_number
image_number = args.image_number
algorithm = args.algorithm
use_mask = True if args.use_mask == 1 else False
print(use_mask)

template_1_url = f"../parts/{part_number}/patch1.jpg"
template_2_url = f"../parts/{part_number}/patch2.jpg"

template_1_img = cv2.imread(template_1_url, cv2.IMREAD_COLOR)
template_2_img = cv2.imread(template_2_url, cv2.IMREAD_COLOR)

template_1_mean, template_1_std_dev = cv2.meanStdDev(template_1_img)
template_2_mean, template_2_std_dev = cv2.meanStdDev(template_2_img)


x_t1, y_t1, _ = template_1_img.shape
x_t2, y_t2, _ = template_2_img.shape
for i in range(20):
    images_url = f"../parts/{part_number}/frame_{image_number}.jpg"
    mask_number = image_number
    if use_mask:
        mask_number = mask_number - 10
    background_url = f"../parts/{part_number}/frame_{mask_number}.jpg"
    background_img = cv2.imread(background_url, cv2.IMREAD_COLOR)
    img = cv2.imread(images_url, cv2.IMREAD_COLOR)
    mask = find_movement(img, background_img)
    algo = create(algorithm, use_mask)
    x1, y1, value1 = algo.run(img, template_1_img, template_1_mean, template_1_std_dev, mask)
    x2, y2, value2 = algo.run(img, template_2_img, template_2_mean, template_2_std_dev, mask)

    output1 = cv2.rectangle(img, (y1, x1), (y1 + y_t1 + 1, x1 + x_t1 + 1), (255, 0, 0), 3)
    output3 = cv2.rectangle(output1, (y2, x2), (y2 + y_t2 + 1, x2 + x_t2 + 1), (0, 255, 0), 3)

    # cv2.imshow('output3', output3)
    # cv2.imshow('template_1_img', template_1_img)
    # cv2.imshow('template_2_img', template_2_img)
    # cv2.imshow('mask', mask)
    # cv2.waitKey()

    cv2.imwrite(f"../res/{part_number}/frame_{image_number}_{algorithm}_{use_mask}_RES_3.jpg", output3)
    image_number = image_number + 10
    print(image_number)



