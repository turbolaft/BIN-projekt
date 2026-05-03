from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class TrainingResult:
    best_val_accuracy: float
    final_train_loss: float


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
) -> TrainingResult:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    final_train_loss = 0.0

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        final_train_loss = running_loss / max(total_samples, 1)
        val_accuracy = evaluate_accuracy(model, val_loader, device)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

    return TrainingResult(best_val_accuracy=best_val_accuracy, final_train_loss=final_train_loss)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, data_loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return correct / max(total, 1)
