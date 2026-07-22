import json
from torchvision import datasets

test_data = datasets.ImageFolder("../data/train")
with open("../models/classes.json", "w") as f:
    json.dump(test_data.classes, f)

print("Classes saved.")