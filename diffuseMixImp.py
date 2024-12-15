import torch
import numpy as np
import pandas as pd
import sklearn
import os
import cv2
from sklearn.model_selection import train_test_split
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch import nn
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import pickle
from lenet import LeNet
from load_data import load_data

import argparse
from torchvision import datasets
from augment.handler import ModelHandler
from augment.utils import Utils
from augment.diffuseMix import DiffuseMix

#from src.main import device

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
#device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

def model_fit(model, opt, lossFn, train_dataloader, val_dataloader):
    #model.to(device)
    # loop over our epochs
    for e in tqdm(range(0, EPOCHS)):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        trainSteps = 0
        valSteps = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        for (x, y) in train_dataloader:
            # send the input to the device
            #x = x.reshape(-1, channel, 128, 128).to(device)
            x = x.reshape(-1, channel, 128, 128)

            # perform a forward pass and calculate the training loss
            #pred = model(x).type(torch.float).squeeze().cpu()
            pred = model(x).type(torch.float).squeeze()

            loss = lossFn(pred, y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainSteps += 1
            pred = torch.round(pred)
            trainCorrect += (pred == y).type(torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in val_dataloader:
                # send the input to the device
                #x = x.reshape(-1, channel, 128, 128).to(device)
                x = x.reshape(-1, channel, 128, 128)

                # make the predictions and calculate the validation loss
                #pred = model(x).type(torch.float).squeeze().cpu()
                pred = model(x).type(torch.float).squeeze()

                totalValLoss += lossFn(pred, y)
                valSteps += 1
                # calculate the number of correct predictions
                pred = torch.round(pred)
                valCorrect += (pred == y).type(torch.float).sum().item()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(train_dataloader.dataset)
        valCorrect = valCorrect / len(val_dataloader.dataset)
        # update our training history
        #H["train_loss"].append(avgTrainLoss.cpu().detach().numpy().item())
        H["train_loss"].append(avgTrainLoss.detach().numpy().item())
        H["train_acc"].append(trainCorrect)

        #H["val_loss"].append(avgValLoss.cpu().detach().numpy().item())
        H["val_loss"].append(avgValLoss.detach().numpy().item())
        H["val_acc"].append(valCorrect)

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))

def plot_roc(p, y):
    fpr, tpr, thresholds = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig("diffMix_roc_curve.png")
    plt.show()


def normal_train():
    channel = 1
    (x0, x1, y0, y1) = load_data()

    x0_train, x0_test, y0_train, y0_test = train_test_split(x0, y0, test_size=0.2, shuffle=True, stratify=y0)
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, shuffle=True, stratify=y1)

    # x_test = x0_test + x1_test
    # y_test = y0_test + y1_test
    # ^^ faced issue x_test = x0_test + x1_test ~~~~~~~~^~~~~~~~~ ValueError: operands could not be broadcast together with shapes (206,128,128) (39,128,128)

    print("Data Loaded: Shape of x0: ", x0.shape, "Shape of y0: ", y0.shape)
    print("Data Loaded: Shape of x1: ", x1.shape, "Shape of y1: ", y1.shape)

    x_test = np.concatenate((x0_test, x1_test), axis=0)
    y_test = np.concatenate((y0_test, y1_test), axis=0)


    '''AUGMENTATION SECTION'''
    # Directory to save the x_train images - later to be used for augmentation
    output_dir = "./x_train/hernia"
    os.makedirs(output_dir, exist_ok=True)

    # Save each image
    for i, img in enumerate(x1_train): # debug - check need to change x1_train to np.array
        filename = os.path.join(output_dir, f"image_{i}.png")
        # Convert image to uint8 (if needed) and save # as used on load_data =>> img = (img / 255.0).astype(np.float32)
        cv2.imwrite(filename, (img * 255).astype(np.uint8))

    augment_dir = './x_train'
    fractal_dir = output_dir

    # Load the dataset (x_train) for augmentation
    train_dataset = datasets.ImageFolder(root=augment_dir)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Load fractal images
    fractal_imgs = Utils.load_fractal_images(fractal_dir)

    # Initialize the model
    model_id = "timbrooks/instruct-pix2pix"
    model_initialization = ModelHandler(model_id=model_id, device='cuda')

    #Prompts
    '''
    Gaussian – Add Gaussian noise with mild intensity.
    Shadow – Add subtle shadows or gradients to simulate lighting variations.
    Contrast – Increase or decrease contrast to simulate different exposure settings.
    Noise – Add general noise (random pixel distortion or interference).
    Grain – Add subtle film grain to simulate low-quality scans.
    Jitter – Apply random brightness, contrast, or saturation changes.
    '''
    prompts = ["slightly bright", "slightly dim"]

    # Create the augmented dataset
    augmented_train_dataset = DiffuseMix(
        original_dataset=train_dataset,
        fractal_imgs=fractal_imgs,
        num_images=1,
        guidance_scale=4,
        idx_to_class=idx_to_class,
        prompts=prompts,
        model_handler=model_initialization
    )

    # Directory to save the augmented images
    aug_dir = "./augmented_images"
    os.makedirs(aug_dir, exist_ok=True)

    for idx, (image, label) in enumerate(augmented_train_dataset):
        image.save(f'augmented_images/{idx}.png')
        #print(f'Image index: {idx}, Label: {label}')
        pass

    '''AUGMENTATION ENDS'''

    '''Load Generated Image'''
    #generated_path = "./result/blended/hernia" #-->> later change to aug_dir?? check for aug_dir
    x_generated = []
    y_generated = []
    for filename in os.listdir(aug_dir): #generated path is aug_dir ->> for filename in os.listdir(generated_path):
        img = cv2.imread(os.path.join(aug_dir, filename), 0) # img = cv2.imread(os.path.join(generated_path, filename), 0)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = (img / 255.0).astype(np.float32)
            # store loaded image
            assert (img.shape == (128, 128))
            x_generated.append(img)
            y_generated.append(1)
    '''Generated image loading ends'''

    # x_train = x0_train + x1_train + x_generated
    # y_train = y0_train + y1_train + y_generated

    x_train = np.concatenate((x0_train, x1_train, x_generated), axis=0)
    y_train = np.concatenate((y0_train, y1_train, y_generated), axis=0)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)

    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    train_dataset = TensorDataset(x_train, y_train)  # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=bsize)  # create your dataloader

    test_dataset = TensorDataset(x_test, y_test)  # create your datset
    test_dataloader = DataLoader(test_dataset, batch_size=bsize)  # create your dataloader
    model = LeNet(numChannels=channel, classes=1)
    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=learingRate)
    lossFn = nn.BCELoss()

    model_fit(model, opt, lossFn, train_dataloader, test_dataloader)

    #model.cpu()
    p = model(x_test.reshape(-1, channel, 128, 128)).squeeze().detach()
    cMat = confusion_matrix(torch.round(p), y_test)
    print(cMat)
    # Save the confusion matrix to a CSV file
    np.savetxt('diffMix_confusion_matrix.csv', cMat, delimiter=',')
    plot_roc(p, y_test)


if __name__ == '__main__':
    bsize = 100
    learingRate = 0.001
    EPOCHS = 5

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    #
    channel = 1
    normal_train()
