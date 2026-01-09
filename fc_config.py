import os
from os.path import join

base_path = os.getenv('NAS_MOUNT_PATH', 'd:/3213/models')
mobile_det = 'PP-OCRv5_mobile_det'
mobile_rec = 'PP-OCRv5_mobile_rec'

oss_config = {
  'endpoint': 'oss-cn-shanghai-internal.aliyuncs.com',
  'bucket': 'whjys',
}
ocr_config = {
  'text_detection_model_dir': join(base_path, 'paddleocr', mobile_det),
  'text_recognition_model_dir': join(base_path, 'paddleocr', mobile_rec),
  'text_detection_model_name' : mobile_det,
  'text_recognition_model_name': mobile_rec,
  'text_rec_score_thresh': 0.9
}
