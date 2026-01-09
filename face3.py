import rotate
import time
import numpy as np
import cv2 as cv
from pathlib import Path


def crop_id(image, faces):
    h,w = image.shape[:2]
    if faces[1] is not None:
        for face in faces[1]:
            coords = face[:-1].astype(np.int32)
            start_row, end_row = coords[1]-round(coords[3]*1), coords[1]+round(coords[3]*2.8)
            start_col, end_col = coords[0]-round(coords[2]*5.2), coords[0]+round(coords[2]*2.5)
            start_row,end_row = 0 if start_row < 0 else start_row, h if end_row > h else end_row
            start_col, end_col = 0 if start_col < 0 else start_col, w if end_col > w else end_col
            roi = image[start_row:end_row, start_col:end_col]
            # height,width = roi.shape[:2]
            # print("身份证区域 w:{} h:{}".format(width, height))
            return roi

count = 0
def visualize(input, faces, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)


def id_card_crop(imagePath, writeFile=False):
    ## [initialize_FaceDetectorYN]
    detector = cv.FaceDetectorYN.create(
        model="model/face_detection_yunet_2023mar.onnx",
        config="",
        nms_threshold=0.5,
        top_k=1,
        score_threshold=0.9,
        input_size=(320, 320)
    )

    # If input is an image
    img1 = cv.imread(cv.samples.findFile(imagePath))
    faces1 = None
    (height, width) = img1.shape[:2]
    scale = 1/round(max(width, height)/1500)
    img1 = cv.resize(img1, (int(width * scale), int(height * scale)))
    
    for i in range(0, 360, 90):
        img1 = rotate.rotate_image(img1, i)
        print(f'angle: {i}')
        (h, w) = img1.shape[:2]
        detector.setInputSize((w, h))
        faces1 = detector.detect(img1)
        if(faces1[1] is not None):
            break
    assert faces1[1] is not None, 'Cannot find a face in {}'.format(imagePath)
    # Draw results on the input image
    # visualize(img1, faces1)
    ccimg = crop_id(img1, faces1)
    global count
    count += 1
    file_name = Path(imagePath).name
    new_file = Path('output') / file_name
    if writeFile:
        cv.imwrite(new_file, ccimg)
        print(f"{count}:write file to {new_file}")
    return ccimg