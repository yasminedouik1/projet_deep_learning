import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from notebooks.models import ImprovedSimpleCNN  # même dossier que app.py

# ======================================================
# PARAMÈTRES — DOIVENT ÊTRE IDENTIQUES AU NOTEBOOK 1
# ======================================================
SR = 16000        # ← était 22050 dans l'ancienne version (BUG #1)
DURATION = 7      # secondes fixes (BUG #2 : absent dans l'ancienne version)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

st.set_page_config(page_title="Baby Cry Classifier", layout="wide")
st.title("🍼 Baby Cry Classification")
st.markdown("### Détection intelligente des pleurs de bébé")

# ──────────────────────────────────────────────────────
# Chargement du modèle
# ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = ImprovedSimpleCNN(num_classes=5)
    checkpoint = torch.load(
        "notebooks/saved_models/ImprovedSimpleCNN.pth",
        map_location="cpu"
    )
    # Gère les deux formats possibles (.pth avec ou sans dict wrapper)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()
class_names = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# ──────────────────────────────────────────────────────
# Fonction de prétraitement — IDENTIQUE au notebook 1
# ──────────────────────────────────────────────────────
def preprocess_audio(audio_path: str) -> torch.Tensor:
    """
    Reproduit exactement le pipeline du notebook 1 :
      1. Chargement à SR=16000 Hz
      2. Padding / truncation à 7 secondes
      3. Mel-spectrogramme (128 mels, hop=512, fft=2048)
      4. Conversion en dB
      5. Normalisation min-max → [0, 1]
      6. Ajout des dimensions batch et canal → (1, 1, 128, T)
    """
    # 1. Chargement à la bonne fréquence
    y, _ = librosa.load(audio_path, sr=SR, duration=DURATION)

    # 2. Padding si l'audio est plus court que DURATION secondes
    target_len = SR * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    y = y[:target_len]

    # 3. Mel-spectrogramme
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )

    # 4. Conversion en dB
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 5. Normalisation min-max (BUG #3 : absente dans l'ancienne version)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # 6. Tensor (1, 1, H, W)
    tensor = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0)
    return tensor, mel_db  # mel_db pour l'affichage

# ──────────────────────────────────────────────────────
# Interface
# ──────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Chargez un fichier audio (.wav)", type=["wav"])

if uploaded_file is not None:
    audio_path = "temp_inference.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file)

    try:
        spec_tensor, spec_db = preprocess_audio(audio_path)
    except Exception as e:
        st.error(f"Erreur lors du prétraitement audio : {e}")
        st.stop()

    # Prédiction
    with torch.no_grad():
        output = model(spec_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item() * 100

    # Affichage des résultats
    col1, col2 = st.columns([1, 1])

    with col1:
        emoji = {"belly_pain": "😖", "burping": "💨", "discomfort": "😣",
                 "hungry": "🍼", "tired": "😴"}
        pred_label = class_names[pred_idx]
        st.success(f"{emoji.get(pred_label, '')} **{pred_label.upper()}** — confiance : {confidence:.1f}%")

        prob_dict = {class_names[i]: float(probs[i]) * 100 for i in range(5)}
        st.bar_chart(prob_dict)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        librosa.display.specshow(
            spec_db, sr=SR, hop_length=HOP_LENGTH,
            x_axis='time', y_axis='mel', ax=ax
        )
        ax.set_title("Mel Spectrogramme (16 kHz, normalisé)")
        plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

    # Détail des probabilités
    with st.expander(" Détail des probabilités par classe"):
        for i, cls in enumerate(class_names):
            st.write(f"**{cls}** : {probs[i].item()*100:.2f}%")
            st.progress(float(probs[i]))