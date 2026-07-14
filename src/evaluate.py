import torch 
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_data = datasets.ImageFolder("../data/test", transform=transformation)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Test image count: {len(test_data)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(test_data.classes))
model.load_state_dict(torch.load("../models/first_model.pt"))
model = model.to(device)
model.eval()

correct_prediction = 0
total_samples = 0

with torch.no_grad():
    for images, tags in test_loader:
        images = images.to(device)
        tags = tags.to(device)
        
        predictions = model(images)
        _, estimated_class = torch.max(predictions, 1)
        correct_prediction += (estimated_class == tags).sum().item()
        total_samples += tags.size(0)
        
test_accuracy = correct_prediction / total_samples
print(f"Test Accuracy: {test_accuracy:.4f}")