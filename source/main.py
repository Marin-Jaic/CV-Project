import os
import threading

import numpy as np
import torch
import torch.nn as nn

from source.data_prep.preprocess import Preprocess
from source.data_prep.utils import compute_class_weights, generate_figures, generate_class_figures, convert, \
    graph_from_pickle
from source.models.UnetV2 import UNetV2
from training import train_model, test_model


def run(unet, epochs, subset_size, num_classes, path, device):

    # -------------------------------------------------------------------------------------------------------------------
    # Training & Testing
    # -------------------------------------------------------------------------------------------------------------------
    use_simple_mask = num_classes == 3

    # A subsample of 1k elements (by default) from the dataset is taken
    train_loader, val_loader, test_loader = Preprocess().get_data_loaders(subset=True, mask_classes=num_classes,
                                                                          subset_size=subset_size,
                                                                          use_simple_mask=use_simple_mask)

    class_weights = compute_class_weights(train_loader, num_classes)

    # Train the model
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_loss, train_iou, train_class_iou, val_loss, val_iou, val_class_iou = train_model(model=unet,
                                                                                           train_loader=train_loader,
                                                                                           val_loader=val_loader,
                                                                                           epochs=epochs,
                                                                                           criterion=criterion,
                                                                                           device=device)

    # Test the model
    test_loss, test_iou, test_class_iou = test_model(model=unet, test_loader=test_loader, device=device)

    # -------------------------------------------------------------------------------------------------------------------
    # Prepare data for plotting
    # -------------------------------------------------------------------------------------------------------------------

    # These values have to be transposed since they have dimension (Epochs, Classes)
    train_class_iou = np.array(train_class_iou).transpose()
    val_class_iou = np.array(val_class_iou).transpose()
    test_class_iou = np.array(test_class_iou).transpose()

    # This is a safe-check since some values are returned as cuda tensors
    # this would in that case convert them to a list of values
    train_loss = convert(train_loss)
    train_iou = convert(train_iou)
    train_class_iou = convert(train_class_iou)

    val_loss = convert(val_loss)
    val_iou = convert(val_iou)
    val_class_iou = convert(val_class_iou)

    # Since it is just 1 value
    test_loss = test_loss.item() if torch.is_tensor(test_loss) else test_loss
    test_iou = test_iou.item() if torch.is_tensor(test_iou) else test_iou
    test_class_iou = convert(test_class_iou)

    # -------------------------------------------------------------------------------------------------------------------
    # Generating Plots
    # -------------------------------------------------------------------------------------------------------------------

    training_data = [("Loss", train_loss), ("IoU", train_iou)]
    validation_data = [("Loss", val_loss), ("IoU", val_iou)]

    generate_figures(path, epochs, num_classes, [training_data, validation_data])
    generate_class_figures(path, epochs, num_classes, [train_class_iou, val_class_iou])

    # for cls in range(len(train_class_iou)):
    #     label = three_class_dict[cls] if num_classes == 3 else all_class_dict[cls]
    #     plt.plot(range(epochs), train_class_iou[cls], label=label)
    #
    # plt.legend()
    # plt.title("Training IoU Per Class")
    # plt.xlabel('Epochs')
    # plt.ylabel('Training IoU')
    # plt.show()
    # plt.savefig(path + f'/figures/{num_classes}-class/Training-Class-IoU.png')
    #
    # for cls in range(len(val_class_iou)):
    #     label = three_class_dict[cls] if num_classes == 3 else all_class_dict[cls]
    #     plt.plot(range(epochs), train_class_iou[cls], label=label)
    #
    # plt.legend()
    # plt.title("Validation IoU Per Class")
    # plt.xlabel('Epochs')
    # plt.ylabel('Training IoU')
    # plt.savefig(path + f'/figures/{num_classes}-class/Training-Class-IoU.png')
    # plt.show()
    # plt.plot(range(epochs), train_loss, label='Training loss')
    # plt.title(f'Training Loss - {num_classes} Class Model')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.savefig(path + f'/figures/{num_classes}-class/Validation-Loss.png')
    # plt.show()
    #
    # plt.plot(range(epochs), torch.tensor(train_iou).cpu(), label='Training IoU')
    # plt.title(f'Training IoU - {num_classes} Class Model')
    # plt.xlabel('Epochs')
    # plt.ylabel('IoU')
    # plt.savefig(path + f'/figures/{num_classes}-class/Training-Mean-IoU.png')
    # plt.show()
    #
    # plt.plot(range(epochs), val_loss, label='Validation loss')
    # plt.title(f'Training Loss - {num_classes} Class Model')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.savefig(path + f'/figures/{num_classes}-class/Validation-Loss.png')
    # plt.show()
    #
    # plt.plot(range(epochs), torch.tensor(val_iou).cpu(), label='Validation IoU')
    # plt.title(f'Training IoU - {num_classes} Class Model')
    # plt.xlabel('Epochs')
    # plt.ylabel('IoU')
    # plt.savefig(path + f'/figures/{num_classes}-class/Validation-Mean-IoU.png')
    # plt.show()


    # -------------------------------------------------------------------------------------------------------------------
    # Save the model weights
    # -------------------------------------------------------------------------------------------------------------------
    # Test Loss: ~0.601, Test IoU: ~0.916 for 18 classes
    # Test Loss: 0.6006672978401184, Test IoU: 0.9173608422279358 for 3 classes
    torch.save(unet.state_dict(), f'trained_models/unet_{epochs}_{subset_size}_{num_classes}.pt')

    # -------------------------------------------------------------------------------------------------------------------
    # Save model results to a text file
    # -------------------------------------------------------------------------------------------------------------------
    filename = f'trained-model-stats/unet_{num_classes}_epochs-{epochs}_size-{subset_size}_test_performance.txt'

    with open(filename, "a") as f:
        print(f'Test Loss: {test_loss:.4f}\n')
        print(f'Test IoU: {test_iou:.4f}\n')
        np.savetxt(f, test_class_iou, delimiter=',', fmt='%.4f', newline='\n', header='Test Class IoU')


if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------------------------------------------------
    epochs = 2
    subset_size = 500

    # -------------------------------------------------------------------------------------------------------------------
    # Auxiliary parameters
    # -------------------------------------------------------------------------------------------------------------------
    path = os.getcwd()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 3-class Model
    # num_classes = 3
    # unet = UNetV2(n_channels=3, n_classes=num_classes).to(device)
    # run(unet, epochs, subset_size, num_classes, path, device)
    #
    # # 18-class Model
    # num_classes = 18
    # unet = UNetV2(n_channels=3, n_classes=num_classes).to(device)
    # run(unet, epochs, subset_size, num_classes, path, device)

    pickle_data = []
    files = []
    n_classes = 18

    for i in range(2):
        filename = f'trained_models/FINAL18class_epoch_{i}_metrics.pkl'
        files.append(filename)

    graph_from_pickle(path, files, n_classes)


