import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os


device = torch.device("cpu")
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("checkpoints/erm_final.pt", map_location=device, weights_only=True))
model.eval()

dfr_head = torch.load("checkpoints/dfr_head.pt", map_location=device, weights_only=True)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


img_path = "artifacts/demo_images/slice_sample_1.jpg"
if not os.path.exists(img_path):
    print(f"Error: Image not found at {img_path}")
    exit()

img = Image.open(img_path).convert("RGB")
tensor_img = transform(img).unsqueeze(0)

with torch.no_grad():
    
    x = model.conv1(tensor_img)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    f = torch.flatten(x, 1)
    
   
    mean = dfr_head["scaler_mean"]
    scale = dfr_head["scaler_scale"]
    coef = dfr_head["coef"]
    intercept = dfr_head["intercept"]
    
    f_norm = (f - mean) / scale
    z = f_norm @ coef + intercept
    
    p_dfr_val = torch.sigmoid(z).item()
    dfr_class = 1 if p_dfr_val > 0.5 else 0
    class_names = ["Landbird", "Waterbird"]
    
    print(f"--- Standalone Verification Results ---")
    print(f"Sample: {img_path}")
    print(f"DFR Predicted Class: {class_names[dfr_class]} (Probability: {p_dfr_val:.4f})")