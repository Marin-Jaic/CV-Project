import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

# Used to map the names
three_class_dict = {0: "Background", 1: "Clothes", 2: "Body"}
all_class_dict = {0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt",
                  6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
                  12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"}


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


def compute_class_weights(train_loader, n_classes, device="cuda"):
    class_counts = torch.zeros(n_classes).to(device)
    for _, mask in train_loader:
        mask = mask.to(device)
        for i in range(n_classes):
            class_counts[i] += torch.sum(mask[:, i, :, :] == 1)
    total_pixels = torch.sum(class_counts)
    class_weights = total_pixels / (n_classes * class_counts)
    return class_weights


# Generate the figures for Training & Validations losses and IoU
def generate_figures(path, epochs, num_classes, results):
    title_list = ["Training", "Validation"]
    for i, x in enumerate(results):
        for data_type, data in x:
            title = title_list[i]
            plt.plot(range(epochs), data, label=f'{title} {data_type}')
            plt.title(f'{title} {data_type} - {num_classes} Class Model')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig(path + f'/figures/{num_classes}-class/{title}-{data_type}.png')
            plt.show()


# Plot the graphs for IoU's per class
def generate_class_figures(path, epochs, num_classes, results):
    title_list = ["Training", "Validation"]

    for i, data in enumerate(results):
        title = title_list[i]
        fig, ax = plt.subplots()

        for cls in range(len(data)):
            label = three_class_dict[cls] if num_classes == 3 else all_class_dict[cls]
            ax.plot(range(epochs), data[cls], label=label)

        # Set handles
        plt.title(f'{title} IoU Per Class')
        plt.xlabel('Epochs')
        plt.ylabel(f'{title} IoU')
        handles, labels = ax.get_legend_handles_labels()

        # Create one new figure for the legend
        fig_legend = plt.figure(figsize=(3, 8))
        axi = fig_legend.add_subplot(111)
        axi.legend(handles, labels, loc='center')
        axi.axis('off')  # Turn off the axis

        fig_legend.canvas.draw()
        fig_legend.savefig(path + f'/figures/{num_classes}-class/Class-IoU-Legend.png')

        plt.savefig(path + f'/figures/{num_classes}-class/{title}-Class-IoU.png')
        plt.show()


# Convert data from torch cuda tensor to list
def convert(data):
    for i in range(len(data)):
        if torch.is_tensor(data[i]):
            data[i] = data[i].item()
        if isinstance(data[i], list):  # In case of `per_class` variables
            data[i] = convert(data[i])
    return data


def plot_from_pickle(path, data_type, epochs, num_classes, results):
    fig, ax = plt.subplots()

    for c in range(len(results[0])):
        label = three_class_dict[c] if num_classes == 3 else all_class_dict[c]
        ax.plot(range(epochs), results[:, c], label=label)

    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title(f'{data_type} per Class')

    handles, labels = ax.get_legend_handles_labels()

    # Create one new figure for the legend
    fig_legend = plt.figure(figsize=(3, 8))
    axi = fig_legend.add_subplot(111)
    axi.legend(handles, labels, loc='center')
    axi.axis('off')  # Turn off the axis

    fig_legend.canvas.draw()
    fig_legend.savefig(path + f'Legend.png')
    plt.show()


def graph_from_pickle(path, files, n_classes):
    ious, dice_scores, accuracies = [], [], []
    epochs = len(files)

    for file in files:
        with open(file, 'rb') as f:
            iou, dice_score, accuracy = pickle.load(f)
            iou = np.array(iou).mean(axis=0)
            ious.append(iou.tolist())
            dice_scores.append(dice_score.tolist())
            accuracies.append(accuracy.tolist())

    ious = np.array(ious)
    accuracies = np.array(accuracies)

    generate_figures(path, epochs, n_classes, [[("Dice Score", dice_scores)]])
    plot_from_pickle(path, 'IoU', epochs, n_classes, ious)
    plot_from_pickle(path, 'Accuracy', epochs, n_classes, accuracies)
