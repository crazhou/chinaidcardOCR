import os
from glob import glob
from img2base64 import image_to_base64
from openai import OpenAI

def alicard_recognize(img_path=None):
  client = OpenAI(
      # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
      api_key=os.getenv("DASHSCOPE_API_KEY"),
      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  )
  completion = client.chat.completions.create(
      model="qwen-vl-ocr-2025-08-28",
      messages=[
          {
              "role": "user",
              "content": [
                  {
                      "type": "image_url",
                      "image_url": image_to_base64(img_path),
                      # 输入图像的最小像素阈值，小于该值图像会按原比例放大，直到总像素大于min_pixels
                      "min_pixels": 28 * 28 * 4,
                      # 输入图像的最大像素阈值，超过该值图像会按原比例缩小，直到总像素低于max_pixels
                      "max_pixels": 28 * 28 * 8192
                  },
                    # qwen-vl-ocr-latest支持在以下text字段中传入Prompt，若未传入，则会使用默认的Prompt：Please output only the text content from the image without any additional descriptions or formatting.
                  # 如调用qwen-vl-ocr-1028，模型会使用固定Prompt：Read all the text in the image.不支持用户在text中传入自定义Prompt
                  {"type": "text",
                  "text": "请提取身份证图像中的姓名、住址、民族、出生日期、身份证号码。要求准确无误的提取上述关键信息、不要遗漏和捏造虚假信息，模糊或者强光遮挡的单个文字可以用英文问号?代替。返回数据格式以json方式输出"},
              ]
          }
      ],
      stream=True,
      stream_options={"include_usage": True}
  )
  full_content = ""
  for chunk in completion:
      if not chunk.choices  or chunk.choices[0].delta.content is None:
          continue
      full_content += chunk.choices[0].delta.content
      # print(chunk.choices[0].delta.content)
  return full_content