from card_correction.card_detection_correction import CardDetectionCorrection
from card_correction.ocr_recognition import OCRRecognitionPipeline
import cv2
import os 
import time
from pathlib import Path
from typing import List, Dict, Any
from face3 import id_card_crop
from textdect import text_dect

# 修改为指向你本地模型所在目录
model_path = Path(os.environ.get('NAS_MOUNT_PATH', "D:/34443")) / "models/cv_resnet18_card_correction"
correct_card = CardDetectionCorrection(model=str(model_path))

model_path = Path(os.environ.get('NAS_MOUNT_PATH', "D:/34443")) / "models/convnext_ocr"
recognizer = OCRRecognitionPipeline(model=str(model_path))


def recognize_card(input_path: str) -> Dict[str, Any]:
     # 找出身份证区域
     img = id_card_crop(input_path)
     # 矫正区域
     img_arr = []
     x_min = 130
     out = correct_card(img)
     if 'output_imgs' in out and len(out['output_imgs']) > 0:
          imgs = out['output_imgs']
          for idx,img in enumerate(imgs):
               rgbimg = img[...,::-1]
               img = cv2.resize(rgbimg, (1200, 762), interpolation=cv2.INTER_LINEAR)
               img, tr, br = text_dect(img, True)
               (x1,y1), (x2, y2) = tr
               if x1 < x_min:
                    x1 = x_min
               img_arr.append(img[y1:y2, x1:x2])
               (x1,y1), (x2, y2) = br
               img_arr.append(img[y1:y2, x1:x2])
          if len(img_arr) == 2:
               r1 = recognizer(img_arr)
               keys = ['name', 'idNumber']
               res = dict(zip(keys, r1['text']))
               return res
     
if __name__ == '__main__':
     files = Path("images").glob("*.jpg")
     for idx,file in enumerate(files):
          start = time.perf_counter()
          res = recognize_card(str(file))   
          end = time.perf_counter()
          print(f"Processing time for {file}: {end - start:.4f} seconds")
          print(f"{idx}识别成功:{file}: {res}") 

               
