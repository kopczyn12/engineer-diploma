import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def train_model(model, criterion, optimizer, train_loader, device, epochs=25):
    """
    Trains the given model with the specified parameters.

    Args:
    model: The model to train.
    criterion: The loss function.
    optimizer: The optimizer to use.
    train_loader: DataLoader for the training data.
    device: The device type ('cpu' or 'cuda').
    epochs: The number of epochs to train the model.

    Returns:
    None
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.logits, 1)
            loss = criterion(outputs.logits, labels)

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    torch.save(model.state_dict(), 'vit.pth')
    print('Training complete')

def evaluate_model(model, dataloader, device):
    """
    Evaluates the given model with the specified dataloader.

    Args:
    model: The model to evaluate.
    dataloader: DataLoader for the evaluation data.
    device: The device type ('cpu' or 'cuda').

    Returns:
    all_preds: The predictions made by the model.
    all_labels: The actual labels for the data.
    """
    model.eval()  # Set model to evaluate mode

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs.logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit the ViT model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize with ImageNet mean and std
])

def load_data(train_dir, test_dir):
    """
    Loads training and test data from the specified directories.

    Args:
    train_dir: The directory containing the training data.
    test_dir: The directory containing the test data.

    Returns:
    train_loader: DataLoader for the training data.
    test_loader: DataLoader for the test data.
    """
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=5, ignore_mismatched_sizes=True)
model.classifier = nn.Linear(model.classifier.in_features, 5)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader, test_loader = load_data('./engineer-diploma/model/data/Dataset_emotions/train',
                                      './engineer-diploma/model/data/Dataset_emotions/valid')

train_model(model, criterion, optimizer, train_loader, device, epochs=25)

# Evaluation
preds, labels = evaluate_model(model, test_loader, device)

print("Classification Report:")
report = classification_report(labels, preds, output_dict=True, zero_division=1)
df = pd.DataFrame(report).transpose()
df.to_csv('results_vit.csv')

confusion_mtx = confusion_matrix(labels, preds)
sns.heatmap(confusion_mtx, annot=True, fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.savefig('confusin_matrix_vit.png')

