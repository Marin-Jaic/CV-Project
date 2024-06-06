import torch
import numpy as np
import torch.nn.functional as F

from matplotlib import pyplot as plt

from source.models.DiceLoss import multiclass_dice_coeff


def compute_metrics(model, test_loader, device, n_classes):
    model.eval()
    ious = []
    total_dice = 0
    total_iou = 0
    accuracies = np.zeros(n_classes)
    with torch.no_grad():
        for images, true_masks in test_loader:
            images, true_masks = images.to(device), true_masks.to(device, dtype=torch.long)
            pred_masks = model(images)
            mask_true = true_masks
            mask_pred = F.one_hot(pred_masks.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            total_dice += multiclass_dice_coeff(pred_masks[:, 1:], true_masks[:, 1:], reduce_batch_first=False)
            # compute the accuracy per class
            for mask_class in range(n_classes):
                # Select only the pixels that are classified as that class in both the predicted and true masks
                mask_pred_class = mask_pred[:, mask_class]
                mask_true_class = mask_true[:, mask_class]
                mask_overlap = mask_pred_class.int() & mask_true_class.int()

                # Compute the accuracy for the selected pixels
                mask_overlap = mask_pred_class.int() & mask_true_class.int()

                # Compute the accuracy for the selected pixels
                if mask_true_class.sum().item() > 0:  # Only compute accuracy for classes that appear in the mask
                    accuracies[mask_class] += (mask_overlap.sum().item() / mask_true_class.sum().item())
            # compute the IoU per class
            intersection = (mask_pred * mask_true).sum(dim=(1, 2, 3))
            union = mask_pred.sum(dim=(1, 2, 3)) + mask_true.sum(dim=(1, 2, 3)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.append(iou.cpu().numpy())
            total_iou += iou.mean().item()
    dice_scores = total_dice / len(test_loader)
    accuracies /= len(test_loader)

    return ious, dice_scores, accuracies


def show_mask_comparison(mask, output):
    n_classes = mask.shape[0]
    fig, ax = plt.subplots(n_classes, 2)
    for i in range(n_classes):
        ax[i, 0].imshow(mask[i])
        ax[i, 1].imshow(output[i])
    plt.show()


def show_mask_comparison_per_class(mask, output, class_idx):
    n_classes = mask.shape[0]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(mask[class_idx])
    ax[1].imshow(output[class_idx])
    plt.show()


def show_output_mask(output):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(output[0])
    plt.show()


def show_image(image, mask):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.permute(0, 1, 2)[0])
    ax[1].imshow(mask[2])
    plt.show()
