import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from tqdm import tqdm
import pickle
from pathlib import Path

def get_optimizer(name, model, lr):
    if name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr)
    elif name == 'Adagrad':
        return optim.Adagrad(model.parameters(), lr=lr)
    elif name == 'RMSProp':
        return optim.RMSprop(model.parameters(), lr=lr)
    elif name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

def train_and_evaluate(optimizer_name, train_loader, test_loader, epochs, device, lr, dropout=False):
    model = CNN(dropout=dropout).to(device)
    optimizer = get_optimizer(optimizer_name, model, lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
    convergence_epoch = None
    log_lines = []

    for epoch in range(epochs):
        epoch_header = f"\nEpoch {epoch+1}/{epochs} - Optimizer: {optimizer_name}, LR: {lr}"
        print(epoch_header)
        log_lines.append(epoch_header)

        model.train()
        correct, total, loss_sum = 0, 0, 0
        for data, target in tqdm(train_loader, desc="Training", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * data.size(0)
            correct += (output.argmax(1) == target).sum().item()
            total += data.size(0)

        train_acc = correct / total
        train_loss = loss_sum / total
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)

        model.eval()
        correct, total, loss_sum = 0, 0, 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating", leave=False):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                loss_sum += loss.item() * data.size(0)
                correct += (output.argmax(1) == target).sum().item()
                total += data.size(0)

        test_acc = correct / total
        test_loss = loss_sum / total
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)

        if convergence_epoch is None and test_acc >= 0.9:
            convergence_epoch = epoch + 1

        log_line = f"[{optimizer_name} | lr={lr}] Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}"
        print(log_line)
        log_lines.append(log_line)

    history["convergence_epoch"] = convergence_epoch or "N/A"

    # 保存训练记录
    record_dir = Path("results/records")
    record_dir.mkdir(parents=True, exist_ok=True)
    save_path = record_dir / f"{optimizer_name}_lr{lr}_dropout{dropout}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(history, f)

    # 保存训练日志
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{optimizer_name}_lr{lr}_dropout{dropout}.log"
    with open(log_file, "w") as f:
        f.write("\n".join(log_lines))

    return history