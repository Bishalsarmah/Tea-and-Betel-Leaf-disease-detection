import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import streamlit as st
from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# =====================================================================
# 1. ADVANCED METRIC-DRIVEN CSS ENGINE
# =====================================================================
def inject_premium_ui_theme():
    st.markdown("""
        <style>
        /* Global Framework Overrides */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background-color: #F8FAFC !important;
            color: #1E293B !important;
        }
        
        [data-testid="stHeader"] { background: transparent; }
        
        /* Modernized Sidebar Configuration */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            box-shadow: 2px 0 12px rgba(0,0,0,0.01);
        }
        
        /* Custom UI Card Element Matrix */
        .premium-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01);
            margin-bottom: 16px;
            transition: transform 0.2s ease;
        }
        .premium-card:hover {
            transform: translateY(-2px);
        }
        
        .card-header-text {
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            color: #64748B !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
        }
        
        /* Flat KPIs Styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            color: #0F172A !important;
            letter-spacing: -0.02em;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            color: #64748B !important;
            text-transform: uppercase;
        }
        
        /* Progress Bars Styling */
        .progress-label-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 500;
            color: #334155;
            margin-bottom: 4px;
        }
        
        /* Streamlit Core Element Adjustments */
        .stProgress > div > div > div > div {
            background-color: #10B981 !important;
        }
        
        /* Custom Table Layout rows */
        .summary-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #F1F5F9;
            font-size: 0.87rem;
        }
        .summary-label { color: #64748B; font-weight: 500; }
        .summary-value { color: #0F172A; font-weight: 600; }
        
        /* Technical Meta Specs Styling */
        .tech-spec-box {
            background: #F1F5F9;
            border-radius: 8px;
            padding: 12px;
            font-family: monospace;
            font-size: 0.78rem;
            color: #475569;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)

# =====================================================================
# 2. MACHINE LEARNING INFRASTRUCTURE CORES
# =====================================================================
class Stage1BinaryFilter(nn.Module):
    def __init__(self):
        super(Stage1BinaryFilter, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_stage1_guardrail():
    model = Stage1BinaryFilter()
    weight_path = "stage1_binary_filter.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    return model

def compute_groups(channels, target_groups):
    if channels % target_groups == 0: return target_groups
    for g in range(min(target_groups, channels), 0, -1):
        if channels % g == 0: return g
    return 1

class ELA_S(nn.Module):
    def __init__(self, inp, kernel_size=5, target_gn_groups=16):
        super(ELA_S, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        pad = kernel_size // 2
        groups = max(1, inp // 8)
        self.conv = nn.Conv1d(inp, inp, kernel_size=kernel_size, padding=pad, groups=groups, bias=False)
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
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_channels != cnf.input_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(cnf.input_channels, cnf.expanded_channels, 1, bias=False),
                norm_layer(cnf.expanded_channels), activation_layer(inplace=True)
            ))
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(nn.Sequential(
            nn.Conv2d(cnf.expanded_channels, cnf.expanded_channels, cnf.kernel, stride, cnf.kernel//2, cnf.dilation, cnf.expanded_channels, bias=False),
            norm_layer(cnf.expanded_channels), activation_layer(inplace=True)
        ))
        if cnf.use_se: layers.append(ELA_S(cnf.expanded_channels))
        layers.append(nn.Sequential(nn.Conv2d(cnf.expanded_channels, cnf.out_channels, 1, bias=False), norm_layer(cnf.out_channels)))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res_connect else self.block(x)

@st.cache_resource
def create_model(num_classes=10):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large")
    return MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel, num_classes=num_classes,
        block=InvertedResidualELA_S, norm_layer=nn.BatchNorm2d
    )

# =====================================================================
# 3. MAPPING LABELS & META SCHEMAS
# =====================================================================
CLASS_NAMES = [
    'betel_bacterial_leaf', 'betel_dried_leaf', 'betel_fungal_brown_spot', 'betel_healthy_leaf',
    'tea_algal_spot', 'tea_brown_blight', 'tea_gray_blight', 'tea_healthy', 'tea_helopeltis', 'tea_red_spot'
]

def get_disease_metadata(class_name):
    if "healthy" in class_name: return "Healthy Matrix", "Optimal", "No immediate counter-agent required."
    elif "bacterial" in class_name: return "Bacterial Pathogen", "High Severity", "Apply agricultural copper oxychloride formulation."
    elif "dried" in class_name: return "Abiotic Stress Factor", "Moderate", "Regulate canopy humidity and shade control."
    else: return "Fungal Pathogen Strain", "Moderate", "Apply broad-spectrum systemic triazole or fungicide."

# =====================================================================
# 4. STREAMLIT CONTROL ROUTER
# =====================================================================
def main():
    st.set_page_config(page_title="Plant Pathology Dashboard", layout="wide", initial_sidebar_state="expanded")
    inject_premium_ui_theme()
    
    # Title Banner Block
    st.markdown("<h2 style='margin-top:-30px; font-weight:700; color:#0F172A; letter-spacing:-0.03em;'>Plant Pathology Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:0.9rem; margin-top:-10px; margin-bottom:24px;'>Multi-stage deep learning framework for agricultural feature validation and diagnosis.</p>", unsafe_allow_html=True)

    # ── STATE A: NO FILE HAS BEEN UPLOADED (UTILIZE EMPTY SPACE) ──
    if "uploaded_specimen_file" not in st.session_state or st.session_state.uploaded_specimen_file is None:
        
        # Large central structural upload container block
        up_col1, up_col2, up_col3 = st.columns([1, 2, 1])
        with up_col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<div class='premium-card' style='text-align:center; padding: 40px 20px;'>", unsafe_allow_html=True)
            st.markdown("<span style='font-size:3rem;'>🍃</span>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-weight:600; margin-top:12px;'>Initialize Optical Diagnostic Scan</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#64748B; font-size:0.85rem; margin-bottom:20px;'>Upload a pristine high-resolution RGB image of a target Tea or Betel crop leaf specimen to execute inference routing.</p>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Upload Leaf Sample Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="file_input_root")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if uploaded_file:
                st.session_state.uploaded_specimen_file = uploaded_file
                st.rerun()

        # Fill horizontal space with a structural architectural workflow map 
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div class='card-header-text' style='text-align:center;'>System Pipeline Topology Blueprint</div>", unsafe_allow_html=True)
        
        wf1, wf2, wf3 = st.columns(3)
        with wf1:
            st.markdown("<div class='premium-card' style='text-align:center; background:#F8FAFC;'><span style='font-weight:600; color:#10B981;'>STAGE 01</span><br><p style='font-size:0.8rem; color:#64748B; margin-top:4px;'>Binary Input Guardrail Filter<br>(Leaf vs Non-Leaf Detection)</p></div>", unsafe_allow_html=True)
        with wf2:
            st.markdown("<div class='premium-card' style='text-align:center; background:#F8FAFC;'><span style='font-weight:600; color:#3B82F6;'>STAGE 02</span><br><p style='font-size:0.8rem; color:#64748B; margin-top:4px;'>MobileNetV3 + ELA_S Attention Block<br>(10-Class Disease Parsing)</p></div>", unsafe_allow_html=True)
        with wf3:
            st.markdown("<div class='premium-card' style='text-align:center; background:#F8FAFC;'><span style='font-weight:600; color:#8B5CF6;'>STAGE 03</span><br><p style='font-size:0.8rem; color:#64748B; margin-top:4px;'>Grad-CAM Visual Explainer Map<br>(Spatial Neural Saliency Extraction)</p></div>", unsafe_allow_html=True)

    # ── STATE B: IMAGE UPLOADED (SPLIT WORKSPACE PROPORTIONATELY) ──
    else:
        uploaded_file = st.session_state.uploaded_specimen_file
        
        # Sidebar Control Unit
        with st.sidebar:
            st.markdown("<div class='card-header-text'>Specimen Source Management</div>", unsafe_allow_html=True)
            st.markdown("<div style='background:white; padding:12px; border-radius:8px; border:1px solid #E2E8F0; font-size:0.8rem; font-weight:500;'>📦 File loaded: " + uploaded_file.name + "</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            top_k = st.slider("Top-K Softmax Distributions", min_value=3, max_value=10, value=5)
            model_path = st.text_input("Core Model Weights", value="mobnetV3_elaS_best_model.pth")
            
            if st.button("Reset Workspace & Clear File", use_container_width=True):
                st.session_state.uploaded_specimen_file = None
                st.rerun()
                
            st.markdown("<br>"*2, unsafe_allow_html=True)
            st.markdown("<div class='card-header-text'>Core Engine Specs</div>", unsafe_allow_html=True)
            st.markdown("""
                <div class='tech-spec-box'>
                    Framework   : PyTorch<br>
                    Backbone    : MobileNetV3-L<br>
                    Attention   : ELA_S Matrix<br>
                    Resolution  : 256 × 256 px
                </div>
            """, unsafe_allow_html=True)

        # Primary Execution Core logic
        image = Image.open(uploaded_file).convert('RGB')
        transform_pipeline = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform_pipeline(image).unsqueeze(0)

        # 🚀 TRACK 1: RUN STAGE 1 VALIDATION GUARDRAIL
        stage1_model = load_stage1_guardrail()
        with torch.no_grad():
            stage1_output = stage1_model(input_tensor)
            stage1_probs = torch.nn.functional.softmax(stage1_output[0], dim=0).cpu().numpy()
            stage1_pred_idx = np.argmax(stage1_probs)

        if stage1_pred_idx == 1:
            st.error("🚨 **Validation Failure: Non-Agricultural Specimen Intercepted**")
            st.warning(f"The structural analysis engine is **{stage1_probs[1]:.2%}** confident that this file does not contain an authentic agricultural leaf structure. Inference loop terminated to safeguard model health.")
            st.image(image, caption="Rejected Sample Array", width=360)
            st.stop()

        # 🚀 TRACK 2: RUN CORE MODEL INFERENCE
        try:
            model = create_model()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()

            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
            inference_ms = (time.time() - start_time) * 1000

            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            raw_class = CLASS_NAMES[pred_idx]
            display_class = raw_class.replace('_', ' ').title()
            
            dis_type, severity, remedy = get_disease_metadata(raw_class)

            # ROW 1: Render 4 High-End Flat Summary KPI Cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(f"<div class='premium-card'>", unsafe_allow_html=True)
                st.metric(label="Target Diagnosis", value=display_class)
                st.markdown("</div>", unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"<div class='premium-card'>", unsafe_allow_html=True)
                st.metric(label="Confidence Matrix", value=f"{confidence:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"<div class='premium-card'>", unsafe_allow_html=True)
                st.metric(label="Classification Type", value=dis_type)
                st.markdown("</div>", unsafe_allow_html=True)
            with kpi4:
                st.markdown(f"<div class='premium-card'>", unsafe_allow_html=True)
                st.metric(label="Severity Tier", value=severity)
                st.markdown("</div>", unsafe_allow_html=True)

            # ROW 2: Horizontal split visualization columns
            v_col1, v_col2 = st.columns(2)
            
            # Extract GradCAM layers
            cam = GradCAM(model=model, target_layers=[model.features[-1]])
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            rgb_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            with v_col1:
                st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-header-text'>Uploaded Specimen Matrix</div>", unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with v_col2:
                st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-header-text'>Neural Attention Saliency (Grad-CAM)</div>", unsafe_allow_html=True)
                st.image(heatmap, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ROW 3: Data logs distribution panel split
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-header-text'>Class Softmax Vector Distribution</div>", unsafe_allow_html=True)
                top_indices = np.argsort(probs)[-top_k:][::-1]
                
                for idx in top_indices:
                    c_name = CLASS_NAMES[idx].replace('_', ' ').title()
                    c_prob = probs[idx]
                    st.markdown(f"""
                        <div class='progress-label-row'>
                            <span>{c_name}</span>
                            <span>{c_prob:.2%}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(c_prob))
                st.markdown("</div>", unsafe_allow_html=True)
                
            with d_col2:
                st.markdown("<div class='premium-card' style='height: 100%;'>", unsafe_allow_html=True)
                st.markdown("<div class='card-header-text'>Diagnostic Action Summary Log</div>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="summary-row"><span class="summary-label">Top Prediction Profile</span><span class="summary-value" style="color:#10B981;">{display_class}</span></div>
                    <div class="summary-row"><span class="summary-label">Hardware Latency Cost</span><span class="summary-value">~{inference_ms:.1f} ms</span></div>
                    <div class="summary-row"><span class="summary-label">Guardrail Verification</span><span class="summary-value" style="color:#10B981;">Passed (Stage 1 Active)</span></div>
                    <div class="summary-row"><span class="summary-label">Attention Layer Mode</span><span class="summary-value">Efficient Local Attention (ELA_S)</span></div>
                    <div class="summary-row" style='border-bottom:none; margin-top:8px;'>
                        <div style='display:flex; flex-direction:column; gap:4px;'>
                            <span class='summary-label'>Clinical Agronomy Action Prescription</span>
                            <span style='color:#0F172A; font-weight:600; font-size:0.95rem; margin-top:4px;'>🧪 {remedy}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Core runtime structural breakdown error: {e}")

if __name__ == "__main__":
    main()