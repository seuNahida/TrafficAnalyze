import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 1. 加载训练好的模型
def load_model(path, device):
    # 加载模型结构
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 6  # 假设类别数为6（包括背景）
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 修改分类头，以适应我们的任务类别数（包括背景类）
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # 加载训练的权重
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model

# 2. 图像预处理
def transform_image(img):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
    ])
    img_tensor = transform(img).unsqueeze(0)  # 增加一个维度以符合模型输入要求
    return img_tensor

# 3. 推理过程，进行检测
def inference(model, img_path, device, threshold=0.5):
    # 加载图像
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform_image(img).to(device)

    with torch.no_grad():  # 推理时不需要梯度
        prediction = model(img_tensor)

    # 提取结果
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # 过滤掉低于阈值的预测
    boxes = boxes[scores > threshold]
    labels = labels[scores > threshold]
    scores = scores[scores > threshold]

    return boxes, labels, scores, img

# 4. 绘制检测结果
def plot_results(boxes, labels, scores, img, class_names):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(np.array(img), aspect='auto')

    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        label = labels[i]
        score = scores[i]

        # 绘制边界框
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='red', linewidth=3))

        # 显示标签和置信度
        ax.text(xmin, ymin, f'{class_names[label]}: {score:.2f}', color='red', fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()

# 5. 主程序
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model_path = 'vehicle_detection.pth'  # 训练模型的路径
    model = load_model(model_path, device)

    # 预定义类别（包括背景）
    class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person', 'background']

    # 手动选择图片进行推理
    img_path = '1.jpg'  # 用户指定的图片路径

    # 推理过程
    boxes, labels, scores, img = inference(model, img_path, device)

    # 绘制检测结果
    plot_results(boxes, labels, scores, img, class_names)

if __name__ == "__main__":
    main()
