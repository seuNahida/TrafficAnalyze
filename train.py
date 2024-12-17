import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from pathlib import Path
from collections import defaultdict
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 1. 自定义数据集类
class VehicleDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted(Path(img_dir).glob('*.jpg'))  # 假设图片为.jpg格式

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = Path(self.label_dir) / (img_path.stem + '.txt')
        
        # 读取图像
        img = Image.open(img_path).convert("RGB")
        img = img.resize((640, 640))  # 确保所有图像大小一致

        # 初始化框和标签列表
        boxes = []
        labels = []

        if os.path.exists(label_path):
            # 读取标签文件
            with open(label_path, 'r') as f:
                for line in f:
                    data = list(map(float, line.split()))
                    label = int(data[0])
                    cx, cy, w, h = data[1:]
                    # 将归一化坐标转换为像素坐标
                    cx, cy, w, h = cx * 640, cy * 640, w * 640, h * 640
                    xmin = cx - w / 2
                    ymin = cy - h / 2
                    xmax = cx + w / 2
                    ymax = cy + h / 2
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)

        # 如果没有标注框，返回空的boxes和labels
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)  # 空的边界框张量
            labels = torch.empty((0,), dtype=torch.int64)      # 空的标签张量

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            img = self.transform(img)

        return img, target

# 2. 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
])

# 3. 获取预训练的Faster R-CNN模型
def get_model(num_classes):
    # 使用预训练的Faster R-CNN模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 修改分类头，以适应我们的任务类别数（包括背景类）
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

# 4. 训练过程
def train(model, dataloader, optimizer, scheduler, num_epochs, device, writer):
    model.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

        for images, targets in progress_bar:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            # 显示损失
            running_loss += losses.item()

            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        scheduler.step()

        # 训练进度显示，估算剩余时间
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (epoch + 1)) * (num_epochs - (epoch + 1))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}, Estimated time left: {remaining_time / 60:.2f} minutes")

        # 记录训练损失到TensorBoard
        writer.add_scalar('Loss/train', running_loss / len(dataloader), epoch)

# 5. 推理过程
def inference(model, img_path, device):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    img = img.resize((640, 640))  # 确保图像尺寸一致
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img)

    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # 绘制边界框和类别标签
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(np.array(img.squeeze().cpu().permute(1, 2, 0)), aspect='auto')

    for i in range(len(boxes)):
        if scores[i] > 0.5:  # 只显示概率高于0.5的框
            xmin, ymin, xmax, ymax = boxes[i]
            label = labels[i]
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color='red', linewidth=3))
            ax.text(xmin, ymin, f'Class {label} ({scores[i]:.2f})', color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()

# 6. 主函数，训练并保存模型
def main():
    # 数据路径
    img_dir = 'images'
    label_dir = 'labels'

    # 检查GPU是否可用，并将设备设置为GPU（如果可用），否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化数据集和数据加载器
    dataset = VehicleDetectionDataset(img_dir, label_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # 初始化模型
    num_classes = 6  # 类别数，包含背景
    model = get_model(num_classes).to(device)

    # 设置优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir='./logs')

    # 训练模型
    train(model, dataloader, optimizer, scheduler, num_epochs=10, device=device, writer=writer)

    # 保存模型
    torch.save(model.state_dict(), 'vehicle_detection.pth')

    # 关闭TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
