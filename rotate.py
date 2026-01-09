import cv2
import numpy as np
import argparse


def rotate_image(image, angle):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    
    # 定义旋转中心（此处为图像中心）
    center = (w // 2, h // 2)
    
    # 生成旋转矩阵
    # 参数：旋转中心，旋转角度，缩放因子（1表示不缩放）
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后的图像尺寸，避免裁剪
    # 计算旋转后的宽高
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    # 新的图像宽度和高度
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵以防止图像被裁剪
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # 应用旋转矩阵进行图像旋转
    # 参数：输入图像，旋转矩阵，输出图像大小
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR,  # 线性插值，保证图像质量
        borderMode=cv2.BORDER_CONSTANT,  # 边界填充方式
        borderValue=(255, 255, 255)  # 边界填充颜色（白色）
    )
    return rotated_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help="输入的图片")
    parser.add_argument('angle', type=int, help="提供一个角度")
    args = parser.parse_args()
    print(f"path: {args.image}, angle:{args.angle}")
    img = cv2.imread(args.image);
    rotate_img = rotate_image(img, args.angle)
    cv2.imshow(f"Rotated by {args.angle} degrees", rotate_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    