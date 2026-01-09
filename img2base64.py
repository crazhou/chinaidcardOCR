import base64
from PIL import Image
import io

def image_to_base64(image_path):
    """
    将图片文件转换为 Base64 编码字符串
    
    参数:
        image_path: 图片文件的路径（如 'test.png'）
    
    返回:
        base64_str: 图片的 Base64 编码字符串
    """
    try:
        # 打开图片文件
        with Image.open(image_path) as img:
            # 创建一个字节流缓冲区
            buffer = io.BytesIO()
            # 将图片保存到缓冲区（格式与原图片一致）
            img.save(buffer, format=img.format)
            # 从缓冲区获取二进制数据
            img_bytes = buffer.getvalue()

            format = img.format.lower()
            # 进行 Base64 编码
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/{format};base64,{base64_str}"
    except Exception as e:
        print(f"转换失败: {e}")
        return None

# 示例用法
if __name__ == "__main__":
    # 替换为你的图片路径
    img_path = "images/id1.jpg"
    base64_code = image_to_base64(img_path)
    
    if base64_code:
        print("图片转换为 Base64 成功！")
        # 打印前 100 个字符（完整字符串可能很长）
        print(f"Base64 编码前 100 字符: {base64_code[:100]}...")
