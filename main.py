import torch
import random
import numpy as np
from custom_loader import load_mnist_from_local
from torch.utils.data import DataLoader, random_split
from train_eval import train_and_evaluate
from utils import plot_curves, print_summary_table

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 超参数
batch_size = 64
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用本地原始 IDX 文件加载 MNIST 数据
full_train, test = load_mnist_from_local('./mnist_data')

# 取前5000训练、前1000测试
train, _ = random_split(full_train, [5000, len(full_train) - 5000])
test, _ = random_split(test, [1000, len(test) - 1000])

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

# 多学习率、多优化器对比
results = {}
for opt in ['SGD', 'Adagrad', 'RMSProp', 'Adam']:
    for lr in [0.01, 0.001]:
        key = f"{opt}_lr{lr}"
        results[key] = train_and_evaluate(opt, train_loader, test_loader, epochs, device, lr)

# Dropout 对比实验
results["Adam_dropout"] = train_and_evaluate("Adam", train_loader, test_loader, epochs, device, 0.001, dropout=True)

# 可视化 + 表格
plot_curves(results)
print_summary_table(results)