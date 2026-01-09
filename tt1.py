from face3 import id_card_crop
from pathlib import Path
import os 
import cv2
import time
from card_correction.card_detection_correction import CardDetectionCorrection
from card_correction.ocr_recognition import OCRRecognitionPipeline

model_path = Path(os.environ.get('NAS_MOUNT_PATH', "D:/34443")) / "models/cv_resnet18_card_correction"
correct_card = CardDetectionCorrection(model=str(model_path))

model_path = Path(os.environ.get('NAS_MOUNT_PATH', "D:/34443")) / "models/convnext_ocr"
recognizer = OCRRecognitionPipeline(model=str(model_path))

if __name__ == "__main__":
  start = time.perf_counter()
  ccimg = id_card_crop("images/id4.jpg", writeFile=False)
  # cv2.imshow("ccimg", ccimg)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  end = time.perf_counter()
  print(f"ID card crop time: {end - start:.4f} seconds")
  start = time.perf_counter()
  out = correct_card(ccimg)
  end = time.perf_counter()
  print(f"ID card correction time: {end - start:.4f} seconds")
  start = time.perf_counter()
  if 'output_imgs' in out and len(out['output_imgs']) > 0:
    imgs = out['output_imgs']
    for idx,img in enumerate(imgs):
      rgbimg = img[...,::-1]
      rgbimg = cv2.resize(rgbimg, (850, 540), interpolation=cv2.INTER_LINEAR)
      name_tr, name_bl = (169,64), (290,120)  # Name box coordinates
      id_tr, id_bl = (294,450), (760,490)  # ID number box coordinates
      imgarr = []
      imgarr.append(rgbimg[name_tr[1]:name_bl[1], name_tr[0]:name_bl[0]])
      imgarr.append(rgbimg[id_tr[1]:id_bl[1], id_tr[0]:id_bl[0]])
      cv2.imwrite('output/name.jpg', imgarr[0])
      cv2.imwrite('output/id.jpg', imgarr[1])

      if len(imgarr) == 2:
          r1 = recognizer(imgarr)
          keys = ['name', 'idNumber']
          res = dict(zip(keys, r1['text']))
          print(f"Recognized data: {res}")

      end = time.perf_counter()
      print(f"OCR recognition time: {end - start:.4f} seconds")
      cv2.rectangle(rgbimg, name_tr, name_bl, (0,0,255), 2)
      cv2.rectangle(rgbimg, id_tr, id_bl, (0,0,255), 2)

      file_name = f"output/corrected_id_{idx}.jpg"
      cv2.imwrite(file_name, rgbimg)
      print(f"Corrected ID card image saved to {file_name}")