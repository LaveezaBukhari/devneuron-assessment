from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import json
from fgsm import Attack

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CHANGED: Use the modern `weights` parameter for loading the model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# FIXED: Added Normalize transform, which is crucial for model accuracy
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# FIXED: A new function to properly convert a normalized tensor back to a displayable image
def tensor_to_pil(tensor):
    """Converts a PyTorch tensor to a PIL image after denormalizing."""
    # The tensor has a batch dimension, so we remove it
    tensor = tensor.squeeze(0)
    
    # Define the inverse of the normalization transform
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_tensor_inv = inv_normalize(tensor)
    
    # Convert tensor to PIL Image
    img_pil = transforms.ToPILImage()(img_tensor_inv)
    return img_pil

def image_to_base64(pil_image):
    """Converts a PIL image to a base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG") # Using PNG for better quality
    return base64.b64encode(buffered.getvalue()).decode()

# Initialize your attack class
fgsm_attack_handler = Attack(model)

@app.post("/attack")
async def perform_attack(epsilon: float = Form(0.1), image: UploadFile = File(...)):
    contents = await image.read()
    input_image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Use the corrected preprocessing pipeline
    input_tensor = preprocess(input_image).unsqueeze(0)
    
    # Get the clean prediction
    output = model(input_tensor)
    clean_pred_idx = output.max(1, keepdim=True)[1][0]
    clean_pred_label = idx2label[clean_pred_idx.item()]

    # Generate the adversarial image
  
    adversarial_tensor = fgsm_attack_handler.run_attack(input_tensor, clean_pred_idx, epsilon)

    # Get the adversarial prediction
    adv_output = model(adversarial_tensor)
    adv_pred_idx = adv_output.max(1, keepdim=True)[1][0]
    adversarial_pred_label = idx2label[adv_pred_idx.item()]
    
    # Convert the adversarial tensor to a proper image and then to base64
    adversarial_pil = tensor_to_pil(adversarial_tensor) # FIXED
    adversarial_image_b64 = image_to_base64(adversarial_pil) # FIXED

    attack_success = clean_pred_label != adversarial_pred_label
    
    return {
        "clean_prediction": clean_pred_label.replace("_", " "),
        "adversarial_prediction": adversarial_pred_label.replace("_", " "),
        "adversarial_image_b64": adversarial_image_b64, # CHANGED: Key name for consistency
        "attack_success": attack_success
    }