from paddleocr import PaddleOCR
from fc_config import ocr_config
import json

def recognitor(input):
    ocr = PaddleOCR(
        **ocr_config,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False) # 更换 PP-OCRv5_mobile 模型
    result = ocr.predict(input)

    keep_keys = ["rec_texts", "rec_scores", "rec_boxes"]


    if(len(result) > 0):
        r1 = result[0].json['res']
        res = {key: r1[key] for key in keep_keys if key in r1}
        return res
    return {}
        
    