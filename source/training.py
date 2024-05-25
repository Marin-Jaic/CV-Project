import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from models.DiceLoss import dice_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')

def compute_iou(mask_pred, target_mask):
    mask_pred = mask_pred.squeeze(1)
    mask_pred = (mask_pred > 0.5).float().bool()
    true_mask = target_mask.bool()

    intersection = (mask_pred & true_mask).float().sum((1, 2))
    union = (mask_pred | true_mask).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean()


def train_model(model, train_loader, epochs=5, learning_rate=1e-5, amp=False, weight_decay=1e-8,
                   gradient_clipping=1.0):
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in tqdm(range(epochs)):
        model.train()
        iou = 0

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

            losses.append(loss.item())

        print(f'Epoch: {epoch}, Loss: {losses[-1] / len(train_loader)}, Accuracy: {iou / len(train_loader)}')

    return losses


def test_model(model, test_loader):
    model.eval()
    loss, iou = 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, true_mask in tqdm(test_loader):
            inputs = inputs.to(device=device, memory_format=torch.channels_last)
            true_mask = true_mask.to(device=device, dtype=torch.float)
            pred_mask = model(inputs).to(device=device)

            loss += criterion(pred_mask, true_mask)
            iou += compute_iou(pred_mask, true_mask)

    print(f'Loss: {loss / len(test_loader)}, Accuracy: {iou / len(test_loader)}')

    return loss / len(test_loader), iou / len(test_loader)
