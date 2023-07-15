import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.metrics import f1_score


# 1. Definicja modelu sieci neuronowej
class CovidClassifier(nn.Module):
    def __init__(self):
        super(CovidClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 62 * 62, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 64 * 62 * 62)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 2. Przygotowanie danych
data_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)

full_dataset = datasets.ImageFolder(
    "C:\\Users\\Jakub\\Desktop\\masters_thesis\\dataset", transform=data_transform
)

train_size = int(0.8 * len(full_dataset))  # 80% danych do treningu
test_size = len(full_dataset) - train_size  # 20% danych do testu

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# 3. Inicjalizacja modelu, funkcji straty i optymalizatora
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CovidClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Trenowanie modelu
num_epochs = 10
train_losses = []
test_losses = []

print("GPU ENEABLED:", torch.cuda.is_available())


for epoch in range(num_epochs):
    train_loss = 0
    test_loss = 0

    model.train()  # Ustawienie modelu na tryb treningowy
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()  # Ustawienie modelu na tryb ewaluacji
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    train_loss /= len(train_loader)
    test_loss /= len(test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
    )

# Rysowanie wykresu strat
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# 5. Zapisanie nauczonego modelu
torch.save(model.state_dict(), "covid_classifier.pt")

# 6. Wczytanie modelu do testowania
model.load_state_dict(torch.load("covid_classifier.pt"))
model.eval()

# 7. Testowanie modelu na danych testowych
correct = 0
total = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Przechowujemy prawdziwe etykiety i przewidziane etykiety, aby obliczyć F1 score
        all_labels.extend(labels.detach().cpu().numpy())
        all_predictions.extend(predicted.detach().cpu().numpy())

print(f"Test Accuracy of the model on test images: {100 * correct / total}%")

# Obliczanie F1 score
f1 = f1_score(all_labels, all_predictions, average="weighted")
print(f"F1 Score of the model on test images: {f1}")
