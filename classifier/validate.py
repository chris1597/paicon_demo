import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from transformers import Dinov2ForImageClassification
from google.colab import drive

data_folder = '/content/drive/My Drive/paicon_data'

# Define transformations (same as in the training script)
transform = transforms.Compose([
    transforms.Resize((518, 518)),  # Resize to a larger size first
    transforms.CenterCrop((518, 518)),  # Center crop to 518x518
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize
])

# Load the complete dataset
full_dataset = datasets.ImageFolder(root=data_folder, transform=transform)

# Load the saved validation indices
val_indices = torch.load('val_indices_fold_5.pt')

# Create the validation subset
val_subset = Subset(full_dataset, val_indices)

# Create data loader for the validation subset
batch_size = 32
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Load the pretrained DINOv2 model
dinov2_model = Dinov2ForImageClassification.from_pretrained('facebook/dinov2-large', num_labels=3)
dinov2_model.load_state_dict(torch.load('best_DinoV2_model_fold_5.pth', map_location=torch.device('cpu')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_model = dinov2_model.to(device)
dinov2_model.eval()

# Define the loss function
criterion = nn.CrossEntropyLoss()

def evaluate_model(data_loader, dataset, model, criterion, device):
    model.eval()
    loss = 0.0
    corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            loss += criterion(outputs, labels).item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    loss /= len(dataset)
    accuracy = corrects.float() / len(dataset)
    return loss, accuracy, all_labels, all_preds

# Evaluate on validation data
val_loss, val_acc, val_labels, val_preds = evaluate_model(val_loader, val_subset, dinov2_model, criterion, device)
print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# Compute confusion matrix for validation data
val_cm = confusion_matrix(val_labels, val_preds)

# Plot and save confusion matrix without displaying
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes, ax=ax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Validation Confusion Matrix')
plt.savefig('validation_confusion_matrix.png')
plt.close(fig)

# Print classification report for validation data
val_report = classification_report(val_labels, val_preds, target_names=full_dataset.classes)
print("Validation Classification Report:\n", val_report)

# Save the classification report to a text file
with open('validation_classification_report.txt', 'w') as f:
    f.write(val_report)
