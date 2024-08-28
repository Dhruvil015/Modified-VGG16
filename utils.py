import os
import shutil
import pandas as pd
import torch
import config
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def move_image_to_folders():
    # Paths
    source_folder = "C:\\Users\\dhruv\\dev\\AI\\Deep Learning\\VGG16-IP\\dataset\\train\\train"
    csv_file = "C:\\Users\\dhruv\\dev\\AI\\Deep Learning\\VGG16-IP\\dataset\\trainLabels.csv"
    destination_folder = "C:\\Users\\dhruv\\dev\\AI\\Deep Learning\\VGG16-IP\\dataset\\train"

    # Read CSV file
    df = pd.read_csv(csv_file)

    labels = df['label'].unique()

    # Create folders for each label
    for label in labels:
        os.makedirs(os.path.join(destination_folder, str(label)), exist_ok=True)

    # Iterate through CSV data and copy images
    for index, row in df.iterrows():
        image_id = row['id']
        label = row['label']
        
        source_path = os.path.join(source_folder, str(image_id) + ".png")
        destination_path = os.path.join(destination_folder, str(label), str(image_id) + ".png")
        
        # Copy image to destination folder
        shutil.move(source_path, destination_path)

def save_checkpoint(filepath, model, optimizer):
    print("=====> Save checkpoint")
    checkpoint = {
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=====> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    

def test_model(loader, model):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        loop = tqdm(loader, leave=True)
        correct = 0
        total = 0
        for idx, (images, labels) in enumerate(loop):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Forward pass
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print("Id-{}, Output-{}".format(idx, outputs))
            print("ID-{}, Label={}, Predicted-{}".format(idx, labels, predicted))
            del images, labels, outputs

        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))

    model.train()  # Set the model back to training mode


def calculate_mean_and_std(dataset_path):
    channels_sum = torch.zeros(3)  # Initialize sum of pixel values for each channel
    channels_squared_sum = torch.zeros(3)  # Initialize sum of squared pixel values for each channel
    num_samples = 0  # Get the number of samples in the dataset

    class_names = os.listdir(dataset_path)
    AllImages = []
    for index, name in enumerate(class_names):
        files = os.listdir(os.path.join(dataset_path, name))
        AllImages += list(zip(files, [name]*len(files)))

    # Iterate over the dataset to compute the sum and squared sum of pixel values for each channel
    for image_file, label in AllImages:
        image = Image.open(os.path.join(os.path.join(dataset_path, label), image_file))

        transform = transforms.ToTensor()
        image = transform(image)

        channels_sum += torch.mean(image, dim=[1, 2])
        channels_squared_sum += torch.mean(image ** 2, dim=[1, 2])
        num_samples += 1

    # Calculate the mean and standard deviation for each channel
    mean = channels_sum / num_samples
    std = torch.sqrt((channels_squared_sum / num_samples) - mean ** 2)

    return mean, std

def imshow(img, title):
    npimg = img.numpy() / 2 + 0.5 # converting the image to to numpy and un-normalise it.
    plt.figure(figsize=(config.BATCH_SIZE, 1))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def show_batch_images(dataloader):
    images, labels = next(iter(dataloader))
    img = torchvision.utils.make_grid(images)
    imshow(img, title=[str(x.item()) for x in labels])

