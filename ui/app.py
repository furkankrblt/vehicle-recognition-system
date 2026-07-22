import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import json

st.title("Vehicle Recognition System")
st.write("Upload a photo of a car, and we'll try to guess its make, model, and year.")

@st.cache_resource
def load_model():
    with open("../models/classes.json", "r") as f:
        classes = json.load(f)
        
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("../models/first_model.pt", map_location="cpu"))
    model.eval()
    return model, classes
model, classes = load_model()

transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

uploaded_file = st.file_uploader("Choose photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded photo", width=300)
    
    tensor = transformation(image).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(tensor)
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        top_5 = torch.topk(probabilities, 5)
        
    st.subheader("Prediction Results:")

    with open("../models/parsed_classes.json", "r") as f:
        parsed_classes = json.load(f)

    for probability, index in zip(top_5.values[0], top_5.indices[0]):
        detail = parsed_classes[index]
        st.write(f"**Brand:** {detail['brand']}  |  **Model:** {detail['model']}  |  **Year:** {detail['year']}  —  %{probability.item()*100:.1f}") 