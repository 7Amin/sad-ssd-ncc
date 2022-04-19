import cv2

file = "0052"
vidcap = cv2.VideoCapture(f'./videos/IMG_{file}.MOV')
success, image = vidcap.read()
count = 0
hold = 300
turn = True

while success:
    if count % 10 == 0 and turn == 0:
        # new_image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"./data/{file}_1/frame_{count}.jpg", image)

    if count % 200 == 0:
        turn = not turn
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

