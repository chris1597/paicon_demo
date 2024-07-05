import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import wandb
from torchvision.models import ResNet101_Weights
from torchvision.utils import save_image
import os
from google.colab import drive
from transformers import Dinov2ForImageClassification, Dinov2Config

# Mount Google Drive
drive.mount('/content/drive')


MODEL_NAME = "DinoV2" # "DinoV2"

resolution = None
if MODEL_NAME == "ResNet":
    resolution = 224
elif MODEL_NAME == "DinoV2":
    resolution = 518
else:
    raise ValueError("Unsupported MODEL_NAME")

# Define transformations for training (with augmentation) and validation (without augmentation)
train_transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.CenterCrop((resolution, resolution)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomGrayscale(p=0.1), 
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), 
])

val_transform = transforms.Compose([
    transforms.Resize((resolution, resolution)), 
    transforms.CenterCrop((resolution, resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# get number of cpu cores
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

# Define the path to the data folder in googöe drive
data_folder = '/content/drive/My Drive/paicon_data'

# Load the raw dataset and the augmented
raw_dataset = datasets.ImageFolder(root=data_folder,transform=val_transform)
augmented_dataset = datasets.ImageFolder(root=data_folder,transform=train_transform)

# Initialize training parameters
batch_size = 32
num_epochs = 50
num_folds = 5
patience = 8

# Prepare k-fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True)

# Lists to store results fo each fold
fold_accuracies = []

# K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(raw_dataset)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Initialize Weights & Biases for the fold
    wandb.init(project=f"cross_val_{MODEL_NAME}", name=f"fold_{fold+1}")

    # Print the file names for the current fold
    train_files = [raw_dataset.samples[idx][0] for idx in train_idx]
    val_files = [raw_dataset.samples[idx][0] for idx in val_idx]
    print(f"Train files for fold {fold+1}: {train_files}")
    print(f"Validation files for fold {fold+1}: {val_files}")
    
    # save the indices for the current fold
    torch.save(train_idx, f'train_indices_fold_{fold+1}.pt')
    torch.save(val_idx, f'val_indices_fold_{fold+1}.pt')

    # Create training and validation subsets, train on augmented data, validate on raw data
    train_subset = Subset(augmented_dataset, train_idx)
    val_subset = Subset(raw_dataset, val_idx)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_cores)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_cores)

    first_batch = next(iter(train_loader))
    images, _ = first_batch
    # Denormalize the images
    def denormalize(tensor, mean, std):
      mean = torch.tensor(mean).reshape(1, 3, 1, 1)
      std = torch.tensor(std).reshape(1, 3, 1, 1)
      tensor = tensor * std + mean
      return tensor
    denormalized_images = denormalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Save the first batch of images as a single image file
    save_image(denormalized_images, f'first_{MODEL_NAME}_fold_{fold+1}.png', nrow=8)


    model = None
    if MODEL_NAME == "DinoV2":
        print("Loading DinoV2 model...")

        # Load the pretrained DinoV2 model for image classification
        model = Dinov2ForImageClassification.from_pretrained('facebook/dinov2-large', num_labels=3)

        # Train just classifier layer, freeze the feature extraction layers 
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
 
    elif MODEL_NAME == "ResNet":
        print("Loading ResNet model...")

        # Load pre-trained ResNet model
        model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

        # Modify the final layer to output 3 classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)

        # Freeze all layers except the last one
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")        

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the optimizer and loss function
    criterion = nn.CrossEntropyLoss()

    if MODEL_NAME == "ResNet":
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    elif MODEL_NAME == "DinoV2":
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")        

    # Define the learning rate scheduler, after 3 epochs of no improvement, reduce the learning rate by a factor of 0.1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Initialize variables to keep track of the best model and early stopping
    best_val_acc = 0.0
    epochs_no_improve = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = None
            if MODEL_NAME == "ResNet":
                outputs = model(inputs)
            elif MODEL_NAME == "DinoV2":    
                outputs = model(inputs).logits
            else:
                raise ValueError(f"Unknown model name: {MODEL_NAME}")    
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # calculate running loss
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_subset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = None
                if MODEL_NAME == "ResNet":
                    outputs = model(inputs)
                elif MODEL_NAME == "DinoV2":    
                    outputs = model(inputs).logits
                else:
                    raise ValueError(f"Unknown model name: {MODEL_NAME}")  
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_subset)
        val_acc = corrects.float() / len(val_subset)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Log metrics to Weights & Biases
        wandb.log({
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        # Step the scheduler
        scheduler.step(val_loss)

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_{MODEL_NAME}_model_fold_{fold+1}.pth')
            print("Model improved. Saving the best model.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered for fold {fold+1}. No improvement for {patience} epochs.")
            break

    fold_accuracies.append(best_val_acc.item())
    print(f"Fold {fold+1} Best Validation Accuracy: {best_val_acc:.4f}")

    # Finish the current W&B run
    wandb.finish()

# Print overall results
print(f"Cross-Validation Results: {fold_accuracies}")
print(f"Average Validation Accuracy: {sum(fold_accuracies)/len(fold_accuracies):.4f}")
