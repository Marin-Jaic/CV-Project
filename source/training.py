import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm


def compute_iou(mask_pred, target_mask):
    mask_pred = mask_pred.squeeze(1)
    mask_pred = (mask_pred > 0.5).float().bool()
    true_mask = target_mask.bool()

    intersection = (mask_pred & true_mask).float().sum((1, 2))
    union = (mask_pred | true_mask).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean()


def compute_class_iou(mask_pred, target_mask):
    per_class_iou = []
    mask_pred = mask_pred.transpose(0, 1)
    target_mask = target_mask.transpose(0, 1)

    for cls in range(len(mask_pred)):
        per_class_iou.append(compute_iou(mask_pred[cls], target_mask[cls]).cpu())

    per_class_iou = np.array(per_class_iou)
    return per_class_iou


def train_one_epoch(model, criterion, optimizer, grad_scaler, gradient_clipping, train_loader, device='cpu'):
    losses, mean_iou = 0, 0
    # Number of classes (Batch x Class x Height x Width)
    per_class_iou = np.zeros(train_loader.dataset.dataset.mask_classes)

    for images, true_mask in tqdm(train_loader):
        optimizer.zero_grad()
        images = images.to(device=device, memory_format=torch.channels_last)
        # Dimensions: (4, 3, 600, 400) == (Batch x Classes x Height x Width)
        true_mask = true_mask.to(device=device, dtype=torch.float)
        # Dimensions: (4, 3, 600, 400) == (Batch x Classes x Height x Width)
        pred_mask = model(images).to(device=device)

        loss = criterion(pred_mask, true_mask)
        mean_iou += compute_iou(pred_mask, true_mask)
        per_class_iou += compute_class_iou(pred_mask, true_mask)

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        losses += loss.item()

    return losses / len(train_loader), mean_iou / len(train_loader), per_class_iou / len(train_loader)


def validation_one_epoch(model, criterion, val_loader, device='cpu'):
    model.eval()
    v_loss, v_iou = 0, 0
    v_per_class_iou = np.zeros(val_loader.dataset.mask_classes)

    with torch.no_grad():
        for images, true_mask in tqdm(val_loader):
            images = images.to(device=device, memory_format=torch.channels_last)
            true_mask = true_mask.to(device=device, dtype=torch.float)
            pred_mask = model(images).to(device=device)

            v_loss += criterion(pred_mask, true_mask)
            v_iou += compute_iou(pred_mask, true_mask)
            v_per_class_iou += compute_class_iou(pred_mask, true_mask)

    return v_loss / len(val_loader), v_iou / len(val_loader), v_per_class_iou / len(val_loader)


def train_model(model, train_loader, val_loader, criterion, epochs=10, learning_rate=1e-5,
                amp=False, weight_decay=1e-8, gradient_clipping=1.0, device="cpu"):

    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    train_losses, train_iou = [], []
    val_loss, val_iou = [], []
    train_class_iou, val_class_iou = [], []

    for _ in tqdm(range(epochs)):
        model.train()
        print('\nTraining...')
        train_loss, t_iou, class_iou = train_one_epoch(model, criterion, optimizer, grad_scaler,
                                                       gradient_clipping, train_loader, device)
        train_losses.append(train_loss)
        train_iou.append(t_iou)
        train_class_iou.append(class_iou)
        print(f'\n\nTrain loss: {train_loss:.4f}\tTrain IoU: {t_iou:.4f}\n\n')

        print('\nValidation...')
        v_loss, v_iou, v_per_class_iou = validation_one_epoch(model, criterion, val_loader, device)

        val_loss.append(v_loss)
        val_iou.append(v_iou)
        val_class_iou.append(v_per_class_iou)

        print(f'\n\nValidation loss: {v_loss:.4f}\tValidation IoU: {v_iou:.4f}\n\n')

    return train_losses, train_iou, train_class_iou, val_loss, val_iou, val_class_iou


def test_model(model, test_loader, device='cpu'):
    model.eval()
    loss, mean_iou = 0, 0
    per_class_iou = np.zeros(test_loader.dataset.mask_classes)
    criterion = nn.CrossEntropyLoss()

    print('\n\nTesting...')
    with torch.no_grad():
        for inputs, true_mask in tqdm(test_loader):
            inputs = inputs.to(device=device, memory_format=torch.channels_last)
            true_mask = true_mask.to(device=device, dtype=torch.float)
            pred_mask = model(inputs).to(device=device)

            loss += criterion(pred_mask, true_mask)
            mean_iou += compute_iou(pred_mask, true_mask)
            per_class_iou += compute_class_iou(pred_mask, true_mask)

    loss, mean_iou = loss / len(test_loader), mean_iou / len(test_loader)
    per_class_iou = per_class_iou / len(test_loader)
    print(f'\n\nTest Loss: {loss}, Test IoU: {mean_iou}')

    return loss, mean_iou, per_class_iou
