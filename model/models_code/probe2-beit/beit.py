import torch
from torch import nn
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, BeitForImageClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def train_model(model, criterion, optimizer, train_loader, device, epochs=25):
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
            logits = outputs.logits
            _, preds = torch.max(logits, 1)
            loss = criterion(logits, labels)

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    torch.save(model.state_dict(), 'beit.pth')
    print('Training complete')

def evaluate_model(model, dataloader, device):
    """
    Evaluates a given model on the provided data.

    :param model: The PyTorch model to evaluate.
    :type model: nn.Module
    :param dataloader: DataLoader for the dataset.
    :type dataloader: torch.utils.data.DataLoader
    :param device: Device (cpu or cuda) where the model and data are placed.
    :type device: torch.device
    :return: List of model predictions and actual labels
    :rtype: list, list
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
    Loads image data from specified directories.

    :param train_dir: Path to the training dataset directory.
    :type train_dir: str
    :param test_dir: Path to the test dataset directory.
    :type test_dir: str
    :return: DataLoaders for the training and test dataset
    :rtype: torch.utils.data.DataLoader, torch.utils.data.DataLoader
    """
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
in_features = model.classifier.in_features  # Gets 'in_features' of the last layer
model.classifier = nn.Linear(in_features, 5)  # Replace the entire 'classifier' layer

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
df.to_csv('results_beit.csv')

confusion_mtx = confusion_matrix(labels, preds)
sns.heatmap(confusion_mtx, annot=True, fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.savefig('confusin_matrix_beit.png')

