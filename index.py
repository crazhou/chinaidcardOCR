# encoding=utf8
import json
import oss2
import logging
import os
from os.path import join
from pathlib import Path
from fc_config import oss_config
from card11 import recognize_card

def handler(event, context):
    logger = logging.getLogger()
    evt = json.loads(event)
    auth = oss2.StsAuth(os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"), os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"), os.getenv("ALIBABA_CLOUD_SECURITY_TOKEN"))
    bucket = oss2.Bucket(auth, oss_config['endpoint'], oss_config['bucket'])
    objName = join('form-sh', evt['objectName'])
    tarName = join(os.getenv('NAS_MOUNT_PATH'), 'idcards', Path(evt['objectName']).name)
    # oss 文件下载到本地
    bucket.get_object_to_file(objName, tarName)
    rest= recognize_card(tarName)
    logger.info("events is %s", event)
    logger.info("put Object %s target: %s ok", objName, tarName)
    return json.dumps(rest, indent=2, ensure_ascii=False)