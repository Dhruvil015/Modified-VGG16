import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, test_model, show_batch_images
from torch.utils.data import DataLoader
from model import VGG_Net
from dataset import myImageDataset, data_loader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

def train_fun(loader, model, optimizer, loss):
    loop = tqdm(loader, leave=True)
    
    for idx, (images, labels) in enumerate(loop):
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        cls_loss = loss(outputs, labels)

        # Backward pass
        cls_loss.backward()
        optimizer.step()

def main():
    print("==> PROCESS STARTED WITH ", config.DEVICE.upper())

    # dataset
    dataset = myImageDataset('dataset/train')
    train_loader = data_loader(dataset, config.BATCH_SIZE, True)

    # VGG model
    vgg = VGG_Net().to(config.DEVICE)

    # Loss Function
    loss = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.SGD(vgg.parameters(), lr=config.LEARNING_RATE, weight_decay = 0.005, momentum = 0.9)

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT, vgg, optimizer, config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        train_fun(train_loader, vgg, optimizer, loss)

        if config.SAVE_MODEL:
            save_checkpoint(config.CHECKPOINT, vgg, optimizer)
    
    # save model
    torch.save(vgg.state_dict(), config.MODEL)

    checkpoint = torch.load(config.CHECKPOINT, map_location=config.DEVICE)
    vgg.load_state_dict(checkpoint['state_dict'])

    test_dataset = myImageDataset('dataset/test')
    test_loader = data_loader(test_dataset, config.BATCH_SIZE, True)
    # After completing the training loop, call the testing function
    test_model(test_loader, vgg)


if __name__ == "__main__":
    main()
