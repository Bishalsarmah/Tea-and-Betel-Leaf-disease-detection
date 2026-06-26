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

# ===================== MODEL ARCHITECTURE =====================
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

@st.cache_resource
def create_model(num_classes=10):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large")
    norm_layer = nn.BatchNorm2d
    model = MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        num_classes=num_classes,
        block=InvertedResidualELA_S,
        norm_layer=norm_layer
    )
    return model

# ===================== STREAMLIT APP =====================
CLASS_NAMES = [
    'betel_Bacterial Leaf Disease', 'betel_Dried Leaf',
    'betel_Fungal Brown Spot Disease', 'betel_Healthy Leaf',
    'tea_algal_spot', 'tea_brown_blight',
    'tea_gray_blight', 'tea_healthy',
    'tea_helopeltis', 'tea_red_spot'
]

def main():
    st.set_page_config(page_title="Leaf Disease Diagnosis", layout="wide")
    st.title("🍃 Tea & Betel Leaf Disease Classifier")
    st.markdown("---")
    
    # Sidebar for model upload
    with st.sidebar:
        st.header("Model Configuration")
        model_path = "mobnetV3_elaS_best_model.pth"
        st.info("Using default model: mobnetV3_elaS_best_model.pth")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Image Upload")
        uploaded_file = st.file_uploader("Choose a leaf image", 
                                       type=["jpg", "jpeg", "png"],
                                       help="Upload clear photo of tea/betel leaf")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if uploaded_file:
            st.header("Analysis Results")
            with st.spinner('Analyzing leaf health...'):
                try:
                    # Load model and process image
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
                    input_tensor = transform(image).unsqueeze(0)
                    
                    # Prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)
                        conf, pred_idx = torch.max(probabilities, 0)
                        conf = conf.item()
                    
                    # Display results
                    st.success(f"**Prediction:** {CLASS_NAMES[pred_idx]}")
                    st.info(f"**Confidence:** {conf:.2%}")

                    # ========== Grad-CAM Visualization ==========
                    st.subheader("Model Attention Heatmap")
                    target_layers = [model.features[-1]]
                    cam = GradCAM(model=model, target_layers=target_layers)
                    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                    grayscale_cam = grayscale_cam[0, :]
                    
                    # Prepare and show Grad-CAM
                    rgb_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    
                    fig1 = plt.figure(figsize=(1, 1))
                    plt.imshow(visualization)
                    plt.axis('off')
                    st.pyplot(fig1)
                    plt.close()

                    # ========== Probability Chart ==========
                    st.subheader("Class Probability Distribution")
                    fig2 = plt.figure(figsize=(8, 4))
                    plt.barh(CLASS_NAMES, probabilities.numpy())
                    plt.xlabel("Probability")
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.markdown("**Troubleshooting Tips:**")
                    st.markdown("1. Ensure uploaded image is clear and properly focused")
                    st.markdown("2. Check model file exists in correct location")
                    st.markdown("3. Verify image shows entire leaf surface")

if __name__ == "__main__":
    main()
