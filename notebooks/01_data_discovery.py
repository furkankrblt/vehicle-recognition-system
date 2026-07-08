import os

data_path = "../data/train"

folders = os.listdir(data_path)
print(f"Number of classes: {len(folders)}")
print(f"\nFirst 10 class:")
for k in folders[:10]:
    number_of_image = len(os.listdir(os.path.join(data_path, k)))
    print(f"  {k}: {number_of_image} image")