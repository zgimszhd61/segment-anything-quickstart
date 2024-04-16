import os
from google.colab import drive
from PIL import Image
import numpy as np
import cv2  # 导入OpenCV库
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# 挂载Google Drive
drive.mount('/content/drive')

# 设置模型文件的下载URL和目标路径
model_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
model_path = '/content/drive/MyDrive/sam_vit_h_4b8939.pth'  # 保存到Google Drive中的路径

# 加载模型
sam_checkpoint = model_path  # 更新为正确的路径
model_type = "vit_h"
device = "cpu"  # 在Colab上通常使用CPU，但如果有GPU可以将其改为"cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 选择一张图片进行分割
image_path = 'images/SDT.jpeg'
image = np.array(Image.open(image_path))
predictor.set_image(image)

# 使用文本提示进行分割
text_prompt = "human"
masks, scores, logits = predictor.predict(point_coords=None, point_labels=None, multimask_output=True)

# 假设我们想要显示得分最高的掩码
highest_score_index = np.argmax(scores)
mask_to_display = masks[highest_score_index]

# 将掩码转换为8位单通道图像，以便于使用OpenCV函数
mask_uint8 = (mask_to_display * 255).astype(np.uint8)

# 使用OpenCV找到掩码的轮廓
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个新的图像数组，用于存放最终结果
result_image = np.copy(image)

# 在原图上绘制轮廓线
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 3)  # 使用绿色线条绘制轮廓

# 显示原图和修改后的图像
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result_image)
plt.title('Image with Object Contours')
plt.axis('off')
plt.show()
