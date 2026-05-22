import librosa
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import os

# ======================================================
# CONFIGURATION
# ======================================================
SR = 16000
DURATION = 7
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

class_names = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# Palette de couleurs
COLORS = {
    "primary": "#4AA3A2",
    "light": "#A7E0E0",
    "accent": "#BED3C3",
    "bg": "#4AA3A2",
    "text": "#FFFFFF"
}

# ======================================================
# MODÈLE
# ======================================================
class ImprovedSimpleCNN(nn.Module):
    def __init__(self, num_classes=5, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================================================
# CHARGEMENT DU MODÈLE
# ======================================================
@st.cache_resource
def load_model():
    model = ImprovedSimpleCNN(num_classes=5)
    try:
        checkpoint = torch.load('./notebooks/saved_models/ImprovedSimpleCNN.pth', map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except:
        return None

model = load_model()

# ======================================================
# CONFIG STREAMLIT + CSS
# ======================================================
st.set_page_config(
    page_title="Baby Cry Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {{
        background-color: {COLORS["bg"]} !important;
        font-family: 'Poppins', sans-serif;
    }}

    .main .block-container {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }}

    h1, h2, h3, h4, p, label {{
        color: {COLORS["text"]} !important;
    }}

    /* Bouton */
    .stButton>button {{
        width: 100%;
        border-radius: 999px;
        height: 3.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        background-color: {COLORS["accent"]} !important;
        color: #1e3a34 !important;
        border: none;
    }}

    .stButton>button:hover {{
        background-color: {COLORS["light"]} !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }}

    /* Focus Color */
    .stButton>button:focus,
    .stButton>button:focus-visible,
    input:focus,
    textarea:focus,
    .stTextInput > div > div > input:focus,
    div[data-baseweb="select"] > div:focus-within,
    .stAudioInput > div:focus-within {{
        outline: 3px solid #A7E0E0 !important;
        box-shadow: 0 0 0 3px rgba(167, 224, 224, 0.35) !important;
        border-color: #A7E0E0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# FONCTIONS
# ======================================================
def preprocess_audio(audio_file):
    y, _ = librosa.load(audio_file, sr=SR, duration=DURATION)
    target = SR * DURATION
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    y = y[:target]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    tensor = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0)
    return tensor

def predict(audio_file):
    if model is None:
        return None, None, None
    spec = preprocess_audio(audio_file)
    with torch.no_grad():
        output = model(spec)
        probabilities = torch.softmax(output, dim=1)[0]
    
    pred_idx = probabilities.argmax().item()
    confidence = probabilities[pred_idx].item() * 100
    return probabilities, pred_idx, confidence

# ======================================================
# INTERFACE
# ======================================================
col_title, col_img = st.columns([7, 3])

with col_title:
    st.title("Baby Cry Classifier")
    st.markdown("""
    <span style='font-size:1.5rem;'>
    Cette application permet de reconnaître la cause probable des pleurs d’un bébé à partir d’un enregistrement audio.<br>
    <span style='font-size:1.1rem;'>
    Téléchargez ou enregistrez un cri, puis laissez l’IA analyser et prédire la raison du pleur parmi cinq catégories : faim, fatigue, douleur au ventre, inconfort ou besoin de roter.<br>
    Simple, rapide et utile pour mieux comprendre les besoins de votre bébé !
    </span>
    </span>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='height:20px;'></div>
    """, unsafe_allow_html=True)
    st.markdown("**Modèle : ImprovedSimpleCNN (90.24% accuracy)**")

with col_img:
    image_path = "assets/images/baby.png"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.1); 
                    height: 160px; border-radius: 20px;
                    display: flex; align-items: center; justify-content: center;
                    color: white; font-size: 1rem;">
            Image non trouvée
        </div>
        """, unsafe_allow_html=True)

# ====================== Enregistrement + Upload ======================
st.markdown("### Enregistrer ou télécharger l'audio")

tab1, tab2 = st.tabs(["Enregistrer avec le micro", "Charger un fichier"])

audio_file = None

with tab1:
    recorded_audio = st.audio_input("Appuyez pour enregistrer")
    if recorded_audio:
        audio_file = recorded_audio

with tab2:
    uploaded_file = st.file_uploader("Déposez un fichier audio .wav", type=["wav"])
    if uploaded_file:
        audio_file = uploaded_file

# ====================== Lecture + Analyse ======================
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    if st.button(" Analyser le cri"):
        with st.spinner("Analyse en cours..."):
            probs, pred_idx, confidence = predict(audio_file)
            
            if probs is not None:
                predicted_class = class_names[pred_idx].replace("_", " ").title()
                
                st.markdown("---")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color: rgba(255,255,255,0.1); 
                                padding: 2.5rem 1.5rem; border-radius: 20px; text-align: center;">
                        <h2 style="margin:0; color:#A7E0E0;">Classe Prédite</h2>
                        <h1 style="margin:1rem 0; color:white; font-size: 3rem;">
                            {predicted_class}
                        </h1>
                        <h3 style="margin:0; color:#BED3C3;">Confiance : {confidence:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Distribution des probabilités**")
                    for i, (name, prob) in enumerate(zip(class_names, probs)):
                        name_display = name.replace("_", " ").title()
                        percentage = float(prob) * 100
                        bar_color = "#A7E0E0" if i == pred_idx else "#BED3C3"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 12px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px; font-weight:500;">
                                <span>{name_display}</span>
                                <span><b>{percentage:.1f}%</b></span>
                            </div>
                            <div style="height:12px; background:rgba(255,255,255,0.15); border-radius:10px; overflow:hidden;">
                                <div style="width:{percentage}%; height:100%; background:{bar_color}; border-radius:10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)