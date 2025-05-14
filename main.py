import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import os
import time
import copy

 
# Set data directory
data_dir = '../flower_dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Data augmentation and normalization for training and validation

data_transforms = transforms.Compose([
    # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
    ### START SOLUTION HERE ###
    # Add five data augmentation methods, Normalizating and Tranform to tensor
    transforms.RandomResizedCrop(224),          # Randomly crop to 224x224
    transforms.RandomHorizontalFlip(),          # Random horizontal flip
    transforms.RandomRotation(20),              # Random rotation within ±20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
    transforms.RandomVerticalFlip(),            # Random vertical flip
    transforms.ToTensor(),                      # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], # Normalize with ImageNet mean and std
                         [0.229, 0.224, 0.225])
    ### END SOLUTION HERE ###
])

# Load training and validation datasets
train_dataset = datasets.ImageFolder(train_dir, data_transforms)
val_dataset = datasets.ImageFolder(val_dir, data_transforms)

# Create DataLoaders for training and validation sets
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Pack DataLoaders into a dictionary for convenience
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Get class names from folder names
class_names = train_dataset.classes

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer to match the number of classes
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function (cross entropy for classification)
criterion = nn.CrossEntropyLoss()

# Define optimizer (SGD with momentum)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Learning rate scheduler: decrease LR by factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Define training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()  # Record training start time
    best_model_wts = copy.deepcopy(model.state_dict())  # Store the best model weights
    best_acc = 0.0  # Best accuracy initialized

    # Move model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loop through multiple epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1} / {num_epochs}')
        print('-' * 30)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0  # Accumulate loss
            running_corrects = 0  # Accumulate number of correct predictions

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # Zero gradients for each batch

                # Only compute gradients in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Forward pass
                    _, preds = torch.max(outputs, 1)  # Get predicted class
                    loss = criterion(outputs, labels)  # Compute loss

                    # Backward pass and optimization only in training phase
                    if phase == 'train':
                        loss.backward()      # Backward pass
                        optimizer.step()     # Update parameters

                # Accumulate loss and correct predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Update learning rate after each training phase
            if phase == 'train':
                scheduler.step()

            # Compute average loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} - Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.4f}')

            # Save the best model based on validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_dir = '../work_dir'    # Creat a directory to save models
                os.makedirs(save_dir, exist_ok=True)   # Save model parameters
                torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))

        print()

    # Print training time and best validation accuracy
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model

# Start training
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)


