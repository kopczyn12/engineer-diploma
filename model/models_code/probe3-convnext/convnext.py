import torch
from torch import nn
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, ConvNextForImageClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def train_model(model, criterion, optimizer, train_loader, device, epochs=25):
    """
    Trains the ConvNext model.

    This function trains the ConvNext model for a specified number of epochs. 
    After training, the model's state is saved for later use.

    :param model: The ConvNext model used for classification.
    :type model: ConvNextForImageClassification
    :param criterion: The loss function used for optimization during training.
    :type criterion: CrossEntropyLoss
    :param optimizer: The optimizer used for updating the weights of the model during training.
    :type optimizer: Adam
    :param train_loader: The DataLoader for the training dataset.
    :type train_loader: DataLoader
    :param device: The device (CPU or GPU) that PyTorch will use for computations.
    :type device: device
    :param epochs: The number of times the learning algorithm will work through the entire training dataset (default is 25).
    :type epochs: int

    :return: None
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

    torch.save(model.state_dict(), 'convnext.pth')
    print('Training complete')

def evaluate_model(model, dataloader, device):

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
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
in_features = model.classifier.in_features  # Gets 'in_features' of the last layer
model.classifier = nn.Linear(in_features, 5)  # Replace the entire 'classifier' layer

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader, test_loader = load_data('/Users/mkopczynski/Desktop/engineer-diploma/model/data/Dataset_emotions/train',
                                      '/Users/mkopczynski/Desktop/engineer-diploma/model/data/Dataset_emotions/valid')

train_model(model, criterion, optimizer, train_loader, device, epochs=25)

# Evaluation
preds, labels = evaluate_model(model, test_loader, device)

print("Classification Report:")
report = classification_report(labels, preds, output_dict=True, zero_division=1)
df = pd.DataFrame(report).transpose()
df.to_csv('results_convnext.csv')

confusion_mtx = confusion_matrix(labels, preds)
sns.heatmap(confusion_mtx, annot=True, fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_convnext.png')

