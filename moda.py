from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import cv2
import glob
import time
dg_recognition = pipeline(Tasks.ocr_recognition, model='iic/cv_convnextTiny_ocr-recognition-general_damo', device="cpu")
card_detection_correction = pipeline(Tasks.card_detection_correction, model='iic/cv_resnet18_card_correction', device="cpu")

count = 0
def getRoi(file, save, show=False):
    result = card_detection_correction(file)
    if result['output_imgs'] is not None:
        imgs = result['output_imgs']
        if(len(imgs) == 1):
            img = np.array(imgs[0])
        height, width = img.shape[:2]
        print(f"Image {width}x{height} ratio: {width/height}")
        img = cv2.resize(img, (850, 540))
        name_img = img[55:110,160:300]
        idnum_img = img[440:490, 290:760]
        cv2.rectangle(img, (160, 55), (300, 110),(0, 255, 0), thickness=2)
        cv2.rectangle(img, (290, 440), (760, 490),(0, 255, 0), thickness=2)
        names = dg_recognition([name_img, idnum_img])
        print(f"name:{names}")
        if save:
            global count
            count+=1
            fileName = file.replace(".jpg", "".join(['_', str(count), ".jpg"])).replace('images', 'output')
            cv2.imwrite(fileName, img)
        if show:
            cv2.imshow("idcardroi", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    for name in glob.glob("output/*.jpg"):
        start_time = time.perf_counter() 
        getRoi(name, True, False)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time  # 执行时间（秒）
        print(f"识别用时 {elapsed_time:.6f} 秒")
