import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets

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

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 5. Zapisanie nauczonego modelu
torch.save(model.state_dict(), "covid_classifier.pt")

# 6. Wczytanie modelu do testowania
model.load_state_dict(torch.load("covid_classifier.pt"))
model.eval()

# 7. Testowanie modelu na danych testowych
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy of the model on test images: {100 * correct / total}%")
