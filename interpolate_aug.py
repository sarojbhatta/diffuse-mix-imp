import torch
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import pickle
from lenet import LeNet
from load_data import load_data


print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def interpolate(x1, x2):
    x = x1
    f = 0.6
    for i in range(128):
        for j in range(128):
            x[i][j] = f * x1[i][j] + (1-f) * x2[i][j]
    return x



def model_fit(model, opt, lossFn, train_dataloader, val_dataloader):
    model.to(device)
    valAcc =0
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
            x = x.reshape(-1, channel, 128, 128).to(device)
            # perform a forward pass and calculate the training loss
            pred = model(x).type(torch.float).squeeze().cpu()
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
                x = x.reshape(-1, channel, 128, 128).to(device)
                # make the predictions and calculate the validation loss
                pred = model(x).type(torch.float).squeeze().cpu()
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
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy().item())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy().item())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))

        if valCorrect > valAcc:
            valAcc = valCorrect
            print("Saving the model")
            with open('interpolate_trained_model.pkl', 'wb') as file:
                pickle.dump(model, file)
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
    plt.show()


def normal_train():
    channel = 1
    (x0, x1, y0, y1) = load_data();

    x = x0 + x1
    y = y0 + y1
    x = np.array(x)
    y = np.array(y)
    print("Data Loaded: Shape of x: ", x.shape, "Shape of y: ", y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

    print("Train Data : Shape of x: ", x_train.shape, "Shape of y: ", y_train.shape)
    print("Test Data : Shape of x: ", x_test.shape, "Shape of y: ", y_test.shape)


    idx1 = y_train == 1
    idx0 = y_train == 0
    xx0 = x_train[idx0]
    xx1 = x_train[idx1]

    x1_aug =[]
    y1_aug = []
    s1 = np.array((xx1).shape[0])
    s0 = np.array((xx0).shape[0])

    for i in range(s1):
        for j in range(1):
            a = int(np.random.uniform(low=0, high=s1, size=None))
            x = interpolate(xx1[i], xx1[a])
            x1_aug.append(x)
            y1_aug.append(1)
        break

    x_train = np.concatenate((x_train,np.array(x1_aug)), axis=0)
    y_train = np.concatenate((y_train, np.array(y1_aug)), axis=0)

    print("Train Data : Shape of x: ", x_train.shape, "Shape of y: ", y_train.shape)
    print("Test Data : Shape of x: ", x_test.shape, "Shape of y: ", y_test.shape)

    x_train = torch.Tensor(x_train)  # transform to torch tensor
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    train_dataset = TensorDataset(x_train, y_train)  # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=bsize)  # create your dataloader
    test_dataset = TensorDataset(x_test, y_test)  # create your datset
    test_dataloader = DataLoader(test_dataset, batch_size=bsize)  # create your dataloader
    model = LeNet(numChannels=channel, classes=1)
    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=laringRate)
    lossFn = nn.BCELoss()

    model_fit(model, opt, lossFn, train_dataloader, test_dataloader)

    model = pickle.load(open('interpolate_trained_model.pkl', 'rb'))
    model.cpu()
    p = model(x_test.reshape(-1, channel, 128, 128)).squeeze().detach()
    print(confusion_matrix(torch.round(p), y_test))
    plot_roc(p, y_test)


if __name__ == '__main__':
    bsize = 100
    laringRate = 0.001
    EPOCHS = 25

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    channel = 1
    normal_train()

    # cv2.imshow("img1", x0[0])
    # cv2.imshow("img2", x1[0])
    # cv2.imshow("img3",interpolate(x1[0],x0[0]))
    # cv2.waitKey(0);
    # cv2.destroyAllWindows();

