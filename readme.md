### 读光轻量身份证识别

- [项目介绍](#介绍)
- [目录说明](#目录)
- [快速开始](#快速开始)
- [技术原理](#原理)
- [性能](#性能测试)
- [支持我](#支持我)
- [关注我](#关注我)

### 介绍
- pytorch + 读光开源模型 实现 随手拍身份证中姓名和身份证号 两个字段的提取
- 支持模糊图片 以及 长姓名识别，支持二次开发
- 基于modelscope 魔搭社区 的 两个模型 进行少量的二次开发

> 模型1： 票证检测矫正模型 [iic/cv_resnet18_card_correction](https://modelscope.cn/models/iic/cv_resnet18_card_correction)

> 模型2： 读光-文字识别-行识别模型-中英 [iic/cv_convnextTiny_ocr-recognition-general_damo](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo)、

## 目录
```
│  alimodel.py     使用阿里百炼提供的OCR模型
│  card11.py       批量处理 images 文件夹下的 jpg 证件图片  
│  environment.yml conda 环境配置文件
│  face3.py        基于人脸识别的证件旋转剪裁方法
│  fc_config.py    paddleocr 模型配置
│  img2base64.py   图片转base64编码
│  index.py        阿里云函数计算入口文件
│  moda.py         魔搭平台方式的调用代码
│  pdd.py          paddleocr 调用示例
│  ppocrdet.py     opencv 开源的 ppocr 文本检测封装
│  readme.md       项目介绍 
│  rotate.py       图片旋转方法 
│  s.yaml          阿里云函数计算 配置文件
│  textdect.py     文字检测配置函数 
│  tt1.py          不使用文本检测测试函数
│      
├─card_correction  读光卡证矫正模型二次开发文件
│  │  card_detection_correction.py
│  │  model_resnet18.py
│  │  outputs.py
│  │  pipeline.py
│  │  table_process.py
│  │  __init__.py
│  │  
│  ├─fileio
│  │  │  file.py
│  │  │  io.py
│  │  │  __init__.py
│  │  │  
│  │  └─format
│  │          base.py
│  │          json.py
│  │          jsonplus.py
│  │          yaml.py
│  │          __init__.py
│  │          
│  ├─ocr_recognition 读光行文字识别模型二次开发
│  │  │  base_model.py
│  │  │  base_torch_model.py
│  │  │  model.py
│  │  │  pipeline.py
│  │  │  preprocessor.py
│  │  │  __init__.py
│  │  │  
│  │  └─modules
│  │      └─ConvNextViT
│  │              convnext.py
│  │              main_model.py
│  │              timm_tinyc.py
│  │              vitstr.py
│  │              __init__.py
│  │              
│  └─utils
│          checkpoint.py
│          config.py
│          constant.py
│          device.py
│          file_utils.py
│          logger.py
│          torch_utils.py
│          __init__.py
│          
├─images  用于存入测试的证件图片和行文字图片
│      0123.png
│      3101.png
│      4304.png
│      5221.png
│      9931.png

│      
├─model  YuNet人脸检测模型 和 文字检测模型
│      face_detection_yunet_2023mar.onnx
│      text_detection_cn_ppocrv3_2023may.onnx
│      text_detection_en_ppocrv3_2023may.onnx
│      
└─output 用于存放经过模型处理后的文件 
```
## 快速开始

1. 安装 miniconda, 打开 [清华镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)，
找到对应的版本进行安装，我的电脑是windows 用的这个安装包：Miniconda3-py312_25.9.1-1-Windows-x86_64.exe
这个版本自带一个 python 3.12, 你不用再单独安装 python 了


2. 新建conda 环境
```
conda env create -f environment.yml
```

3. 下载模型文件
我把 padddleocr 和 两个读光模型，打包成压缩文件了，[戳这里下载](https://whjys.oss-cn-shanghai.aliyuncs.com/models.zip)
下载后，在电脑上找一个位置，解压到 models 目录

4. 修改配置
修改 card11.py 文件 修改 12和15行的 model_path 指向你的 models 目录

5. 准备证件图
由于证件比较特殊，请你自己拍一张或多张身份证的正面图，尺寸最好不要超过 1920x1080
命名为 xxxx.jpg 放在 images 文件夹

6. 运行测试
```
conda activate torch-env
python card11.py 
```
如果一切正常的话，你会在命令行看到 输出的人名和证件号
## 原理
1. 先使用矫正模型提取图片中的身份证，模型支持任意角度的身份证 再缩放至 850x540 大小
2. 由于尺寸固定了，可以按位置关系 取名字，身份证号字段的文本框，无需调用文字识别算法
3. 把对应的字段图片送入模型2 进行文字识别即可输出结果 
> 更多细节， 可以查看这篇文章 <a target="_blank" rel="noopener noreferrer"  href="https://mp.weixin.qq.com/s/nfImS0Y2VCdxccnGJTPasg">公众号文章</a>

## 性能测试
暂时还进行专业的性能测试，我在的电脑上粗略测试了10张证件图
| 项目| 说明|
|-----|:------------------|
|cpu   |Intel Core Ultra 9 285H  16核心|
内存| 16GB 8533MT/s|
|输入图片|1920x1080 手机拍摄|
|单张耗时|0.5 - 0.8s |
欢迎提供更多性能数据

## 支持我
<p align="center">
    <img width="180" src="
    https://img-oss.whkdshop.com/wechat4.jpg" alt="光谷东程序人生">
</p>
<p align="center">
  项目开发不易，感谢你的打赏
</p>

## 关注我
<p align="center">
    <img width="180" src="
    https://img-oss.whkdshop.com/mpqrcode.jpg" alt="光谷东程序人生">
</p>
<p align="center">
  关注我的公众号，更新更多技术文章
</p>
