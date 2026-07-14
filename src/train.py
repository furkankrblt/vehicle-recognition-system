import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

import torch.optim as optim

transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

training_data = datasets.ImageFolder("../data/train", transform=transformation)
print(f"Sum number of images: {len(training_data)}")
print(f"Number of class: {len(training_data.classes)}")
print(f"First 3 class: {training_data.classes[:3]}")


training_loader = DataLoader(training_data, batch_size=32, shuffle=True)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(training_data.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"The model is ready. The device it works with.: {device}")


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_count = 25
for epoch in range(epoch_count):
    model.train()
    total_loss = 0.0
    correct_prediction = 0
    total_samples = 0
    
    for images, tags in training_loader:
        images = images.to(device)
        tags = tags.to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, tags)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, estimated_class = torch.max(predictions, 1)
        correct_prediction += (estimated_class == tags).sum().item()
        total_samples += tags.size(0)
        
    average_loss = total_loss / len(training_loader)
    accuracy = correct_prediction / total_samples
    
    print(f"Epoch {epoch+1}/{epoch_count} | Loss: {average_loss:.4f} | Accuracy:   {accuracy:.4f}")
    
torch.save(model.state_dict(), "../models/first_model.pt")
print("Model saved: models/first_model.pt")