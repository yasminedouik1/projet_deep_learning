import librosa
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

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
# CHARGEMENT DU MODÈLE (sans notification)
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
# CONFIG STREAMLIT
# ======================================================
st.set_page_config(
    page_title="Baby Cry Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ====================== CSS MODERNE ======================
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

    h1, h2, h3, h4, .stMarkdown, p, label {{
        color: {COLORS["text"]} !important;
    }}

    /* Bouton Analyse */
    .stButton>button {{
        width: 100%;
        border-radius: 999px;
        height: 3.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        background-color: {COLORS["accent"]} !important;
        color: #1e3a34 !important;
        border: none;
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        background-color: #A7E0E0 !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }}

    /* Supprimer les notifications par défaut */
    .stSuccess, .stInfo, .element-container div[data-testid="stAlert"] {{
        display: none !important;
    }}

    /* Amélioration des progress bars */
    .stProgress > div > div {{
        background-color: {COLORS["light"]} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# PREPROCESSING & PREDICTION
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
    st.markdown("**Modèle : ImprovedSimpleCNN (90.24% accuracy)**")

with col_img:
    try:
        st.image("assets/baby.png", use_column_width=True)
    except:
        st.empty()

uploaded_file = st.file_uploader("Déposez un fichier audio **.wav**", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button(" Analyser le cri"):
        with st.spinner("Analyse en cours..."):
            probs, pred_idx, confidence = predict(uploaded_file)
            
            if probs is not None:
                predicted_class = class_names[pred_idx].replace("_", " ").title()
                
                # === Résultat demandé ===
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color: rgba(255,255,255,0.1); 
                                padding: 2rem; border-radius: 20px; text-align: center;">
                        <h2 style="margin:0; color:{COLORS['light']}">Classe Prédite</h2>
                        <h1 style="margin:0.5rem 0; color:white; font-size: 2.8rem;">
                            {predicted_class}
                        </h1>
                        <h3 style="margin:0; color:{COLORS['accent']}">Confiance : {confidence:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Distribution des probabilités**")
                    for i, (name, prob) in enumerate(zip(class_names, probs)):
                        name_display = name.replace("_", " ").title()
                        percentage = float(prob) * 100
                        
                        # Mettre en évidence la classe prédite
                        bar_color = COLORS["light"] if i == pred_idx else COLORS["accent"]
                        st.markdown(f"""
                        <div style="margin-bottom: 8px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                                <span>{name_display}</span>
                                <span><b>{percentage:.1f}%</b></span>
                            </div>
                            <div style="height:10px; background:rgba(255,255,255,0.15); border-radius:10px; overflow:hidden;">
                                <div style="width:{percentage}%; height:100%; background:{bar_color}; border-radius:10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)