import os
import shutil
from ultralytics import YOLO
import cv2
import torch
print(torch.cuda.is_available())
conf = 0.8
framesskip = 5
testmode = 0


def ProduceImg(videopath):
    video = cv2.VideoCapture(videopath)
    ret, frame = video.read()
    model1 = YOLO("ModelV5.pt")
    count = framesskip
    ids = []
    while ret:
        result1 = model1.track(frame, persist=True, conf=0.8)
        for box in result1[0].boxes:
            x = box.xyxy.tolist()
            if testmode == 1:
                frame = cv2.putText(frame, str(box.conf), (int(x[0][0]), int(x[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 1, cv2.LINE_AA)
                frame = cv2.rectangle(frame, (int(x[0][0]), int(x[0][1])), (int(x[0][2]), int(x[0][3])), (0, 255, 0), 1)
            id = int(box.id)
            if id > len(ids):
                ids.append([box.conf, result1[0].orig_img[int(x[0][1]):int(x[0][3]), int(x[0][0]):int(x[0][2])]])
            else:
                if box.conf > ids[id - 1][0]:
                    ids[id - 1][0] = box.conf
                    ids[id - 1][1] = result1[0].orig_img[int(x[0][1]):int(x[0][3]), int(x[0][0]):int(x[0][2])]
        if testmode == 1:
            try:
                frame = cv2.resize(frame, (700, 700))
                cv2.imshow("frame", frame)
            except():
                continue

        count += framesskip
        video.set(cv2.CAP_PROP_POS_FRAMES, count)
        ret, frame = video.read()
        cv2.waitKey(2)

    for i in range(len(ids)):
        print(str(ids[i][0]))
        cv2.imwrite(str(i+1)+"-"+videopath[:-4]+'.jpg', ids[i][1])

ProduceImg("1_2750_20230911_111935246.avi")
