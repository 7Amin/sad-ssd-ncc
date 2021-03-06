# Creating Video from Images using OpenCV-PythonPython
import cv2
import os

image_folder = '/content'
video_name = 'video.mp4'

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
