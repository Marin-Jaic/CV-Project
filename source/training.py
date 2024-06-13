import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from models.DiceLoss import dice_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_iou(mask_pred, target_mask):
    mask_pred = mask_pred.squeeze(1)
    mask_pred = (mask_pred > 0.5).float().bool()
    true_mask = target_mask.bool()

    intersection = (mask_pred & true_mask).float().sum((1, 2))
    union = (mask_pred | true_mask).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean()


def train_one_epoch(model, criterion, optimizer, grad_scaler, gradient_clipping, train_loader):
    losses, iou = 0, 0

    for images, true_mask in tqdm(train_loader):
        optimizer.zero_grad()
        images = images.to(device=device, memory_format=torch.channels_last)
        # Dimensions: (4, 3, 600, 400) == (Batch x Classes x Height x Width)
        true_mask = true_mask.to(device=device, dtype=torch.float)
        # Dimensions: (4, 3, 600, 400) == (Batch x Classes x Height x Width)
        pred_mask = model(images).to(device=device)

        loss = criterion(pred_mask, true_mask)
        iou += compute_iou(pred_mask, true_mask)
        # iou += dice_loss(pred_mask, true_mask, multiclass=True)

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        losses += loss.item()

    return losses / len(train_loader), iou / len(train_loader)


def train_model(model, train_loader, val_loader, epochs=10, learning_rate=1e-5, amp=False, weight_decay=1e-8,
                gradient_clipping=1.0):
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    train_losses, train_iou = [], []
    val_loss, val_iou = [], []

    for _ in tqdm(range(epochs)):
        model.train()
        print('\nTraining...')
        train_loss, t_iou = train_one_epoch(model, criterion, optimizer, grad_scaler,
                                            gradient_clipping, train_loader)
        train_losses.append(train_loss)
        train_iou.append(t_iou)
        print(f'\n\nTrain loss: {train_loss:.4f}\tTrain IoU: {t_iou:.4f}\n\n')

        print('\nValidation...')
        model.eval()
        v_loss, v_iou = 0, 0

        with torch.no_grad():
            for images, true_mask in tqdm(val_loader):
                images = images.to(device=device, memory_format=torch.channels_last)
                true_mask = true_mask.to(device=device, dtype=torch.float)
                pred_mask = model(images).to(device=device)
                v_loss += criterion(pred_mask, true_mask)
                v_iou += compute_iou(pred_mask, true_mask)

            val_loss.append(v_loss / len(val_loader))
            val_iou.append(v_iou / len(val_loader))

        print(f'\n\nValidation loss: {v_loss / len(val_loader):.4f}\tValidation IoU: {v_iou / len(val_loader):.4f}\n\n')

    return train_losses, train_iou, val_loss, val_iou


def test_model(model, test_loader):
    model.eval()
    loss, iou = 0, 0
    criterion = nn.CrossEntropyLoss()

    print('\n\nTesting...')
    with torch.no_grad():
        for inputs, true_mask in tqdm(test_loader):
            inputs = inputs.to(device=device, memory_format=torch.channels_last)
            true_mask = true_mask.to(device=device, dtype=torch.float)
            pred_mask = model(inputs).to(device=device)

            loss += criterion(pred_mask, true_mask)
            iou += compute_iou(pred_mask, true_mask)

    print(f'\n\nTest Loss: {loss / len(test_loader)}, Test IoU: {iou / len(test_loader)}')

    return loss / len(test_loader), iou / len(test_loader)
