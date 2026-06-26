import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidualConfig, _mobilenet_v3_conf
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ===================== 1. MODEL ARCHITECTURE (MUST MATCH TRAINING) =====================
def compute_groups(channels, target_groups):
    if channels % target_groups == 0:
        return target_groups
    for g in range(min(target_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1

class ELA_S(nn.Module):
    def __init__(self, inp, kernel_size=5, target_gn_groups=16):
        super(ELA_S, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        pad = kernel_size // 2
        groups = max(1, inp // 8)
        self.conv = nn.Conv1d(inp, inp, kernel_size=kernel_size,
                            padding=pad, groups=groups, bias=False)
        self.gn = nn.GroupNorm(compute_groups(inp, target_gn_groups), inp)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x).view(b, c, h)
        x_w = self.pool_w(x).view(b, c, w)
        a_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
        a_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)
        return x * a_h * a_w

class InvertedResidualELA_S(nn.Module):
    def __init__(self, cnf, norm_layer):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_channels != cnf.input_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(cnf.input_channels, cnf.expanded_channels, 1, bias=False),
                norm_layer(cnf.expanded_channels),
                activation_layer(inplace=True)
            ))
        
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(nn.Sequential(
            nn.Conv2d(cnf.expanded_channels, cnf.expanded_channels, cnf.kernel,
                      stride, cnf.kernel//2, cnf.dilation, cnf.expanded_channels, bias=False),
            norm_layer(cnf.expanded_channels),
            activation_layer(inplace=True)
        ))
        
        if cnf.use_se:
            layers.append(ELA_S(cnf.expanded_channels))
            
        layers.append(nn.Sequential(
            nn.Conv2d(cnf.expanded_channels, cnf.out_channels, 1, bias=False),
            norm_layer(cnf.out_channels)
        ))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res_connect else self.block(x)

def create_model(num_classes=10):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large")
    norm_layer = nn.BatchNorm2d  # Change if you used different params
    model = MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        num_classes=num_classes,
        block=InvertedResidualELA_S,
        norm_layer=norm_layer
    )
    return model

# ===================== 2. INFERENCE TEST FUNCTION =====================
def test_inference(image_path, model_path):
    # Load model
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        conf, class_idx = torch.max(probabilities, 0)
    
    # Get class names
    class_names = [
        'betel_Bacterial Leaf Disease', 'betel_Dried Leaf',
        'betel_Fungal Brown Spot Disease', 'betel_Healthy Leaf',
        'tea_algal_spot', 'tea_brown_blight',
        'tea_gray_blight', 'tea_healthy',
        'tea_helopeltis', 'tea_red_spot'
    ]
    
    # ========== Grad-CAM Visualization ==========
    target_layers = [model.features[-1]]  # Last convolutional layer
    
    # Create Grad-CAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate CAM mask
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # Prepare visualization
    rgb_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Grad-CAM Visualization
    plt.subplot(1, 3, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    
    # Class Probabilities
    plt.subplot(1, 3, 3)
    plt.barh(class_names, probabilities.numpy())
    plt.title("Class Probabilities")
    plt.tight_layout()
    plt.show()
    
    # Show results
    print(f"\nPrediction: {class_names[class_idx]} (Confidence: {conf.item():.2%})")

# ===================== 3. RUN THE TEST =====================
if __name__ == "__main__":
    # Configuration
    TEST_IMAGE_PATH = "Bacterial_Leaf_Spot_Disease(11)_aug_2.jpg"  # Replace with your test image
    MODEL_PATH = "mobnetV3_elaS_best_model.pth"
    
    try:
        test_inference(TEST_IMAGE_PATH, MODEL_PATH)
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check model architecture matches training code")
        print("2. Verify model file path and weights")
        print("3. Ensure test image exists and is RGB format")
        print("4. Check normalization layer parameters")