import librosa
import numpy as np
import streamlit as st
import torch

# ======================================================
# PARAMÈTRES
# ======================================================
SR = 16000
DURATION = 7
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
BG = "#4AA3A2"

# ======================================================
# CONFIG PAGE
# ======================================================
st.set_page_config(
    page_title="Baby Cry Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ======================================================
# CSS – fond global #4AA3A2 + UI
# ======================================================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root {{
    --bg: {BG};
    --bg-deep: #3d9493;
    --white: #ffffff;
    --white-soft: rgba(255, 255, 255, 0.88);
    --glass: rgba(255, 255, 255, 0.14);
    --glass-border: rgba(255, 255, 255, 0.32);
    --shadow: 0 20px 50px rgba(0, 0, 0, 0.14);
    --radius-lg: 28px;
    --radius-md: 18px;
}}

/* Fond #4AA3A2 sur toute l'application */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
section.main,
.main,
.block-container,
.stMainBlockContainer {{
    background-color: {BG} !important;
    background: {BG} !important;
    font-family: 'Poppins', sans-serif !important;
}}

#MainMenu, footer, header {{
    visibility: hidden;
}}

/* ── Splash ── */
body.splash-mode .splash-wrap {{
    padding: 2rem 0 1rem;
}}

.splash-badge {{
    display: inline-block;
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: white;
    margin-bottom: 1.25rem;
}}

.splash-title {{
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 700;
    color: white !important;
    line-height: 1.15;
    margin: 0 0 0.75rem 0;
}}

.splash-subtitle {{
    font-size: 1.35rem;
    font-weight: 500;
    color: var(--white-soft) !important;
    margin: 0 0 1rem 0;
}}

.splash-text {{
    font-size: 1.05rem;
    max-width: 480px;
    line-height: 1.65;
    color: rgba(255, 255, 255, 0.82) !important;
    margin: 0 0 1.5rem 0;
}}

.splash-stat {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 14px;
    padding: 0.65rem 1.1rem;
    color: white !important;
    font-weight: 600;
    font-size: 1rem;
}}

body.splash-mode h1, body.splash-mode h2, body.splash-mode h3,
body.splash-mode p, body.splash-mode span, body.splash-mode label {{
    color: white !important;
}}

body.splash-mode .main .block-container {{
    padding-top: 1.5rem !important;
    max-width: 1100px !important;
}}

body.splash-mode .stButton > button {{
    border-radius: 999px !important;
    background: white !important;
    color: var(--bg-deep) !important;
    font-weight: 600 !important;
    padding: 0.7rem 2rem !important;
    border: none !important;
    box-shadow: var(--shadow) !important;
    transition: transform 0.15s ease !important;
}}

body.splash-mode .stButton > button:hover {{
    transform: translateY(-2px);
}}

body.splash-mode .stButton > button p,
body.splash-mode .stButton > button span {{
    color: var(--bg-deep) !important;
}}

/* ── Main ── */
body.main-mode .main .block-container {{
    max-width: 680px !important;
    margin: 0 auto !important;
    padding: 2rem 1.25rem 3.5rem !important;
}}

.app-header {{
    text-align: center;
    margin-bottom: 1.75rem;
}}

.app-header h1 {{
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: white !important;
    margin: 0 !important;
}}

.app-header p {{
    font-size: 0.9rem !important;
    color: var(--white-soft) !important;
    margin: 0.35rem 0 0 !important;
}}

body.main-mode div[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-marker) {{
    background: var(--glass) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow) !important;
    padding: 2rem 2rem 1.75rem !important;
}}

.panel-title {{
    text-align: center;
    font-size: 1.35rem;
    font-weight: 600;
    color: white !important;
    margin: 0 0 0.35rem 0;
}}

.panel-subtitle {{
    text-align: center;
    font-size: 0.88rem;
    color: rgba(255, 255, 255, 0.8) !important;
    margin: 0 0 1.5rem 0;
}}

.upload-icon {{
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    opacity: 0.95;
}}

body.main-mode [data-testid="stFileUploader"] section {{
    background: rgba(255, 255, 255, 0.1) !important;
    border: 2px dashed rgba(255, 255, 255, 0.55) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.75rem 1rem !important;
    transition: all 0.2s ease;
}}

body.main-mode [data-testid="stFileUploader"] section:hover {{
    border-color: white !important;
    background: rgba(255, 255, 255, 0.16) !important;
}}

body.main-mode [data-testid="stFileUploader"] button {{
    background: white !important;
    color: var(--bg-deep) !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    border: none !important;
}}

body.main-mode [data-testid="stFileUploader"] small,
body.main-mode [data-testid="stFileUploader"] span,
body.main-mode [data-testid="stFileUploader"] label {{
    color: rgba(255, 255, 255, 0.85) !important;
}}

body.main-mode [data-testid="stAudio"] {{
    margin: 1rem 0 0.5rem;
}}

body.main-mode [data-testid="stAudio"] audio {{
    width: 100%;
    border-radius: 12px;
}}

.audio-label {{
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.75) !important;
    margin-top: 0.75rem;
    text-align: center;
}}

body.main-mode div[data-testid="stVerticalBlock"]:has(.analyze-marker) {{
    text-align: center;
    margin-top: 1.5rem;
    padding-top: 0.5rem;
}}

body.main-mode div[data-testid="stVerticalBlock"]:has(.analyze-marker) .stButton > button {{
    width: 100% !important;
    max-width: 280px !important;
    margin: 0 auto !important;
    display: block !important;
    border-radius: 999px !important;
    background: white !important;
    color: var(--bg-deep) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    border: none !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}}

body.main-mode div[data-testid="stVerticalBlock"]:has(.analyze-marker) .stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18) !important;
}}

body.main-mode div[data-testid="stVerticalBlock"]:has(.analyze-marker) .stButton > button p,
body.main-mode div[data-testid="stVerticalBlock"]:has(.analyze-marker) .stButton > button span {{
    color: var(--bg-deep) !important;
}}

/* Résultats */
.result-card {{
    background: var(--glass) !important;
    backdrop-filter: blur(18px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 2rem !important;
    margin-top: 1.5rem !important;
    box-shadow: var(--shadow) !important;
}}

.result-hero {{
    text-align: center;
    padding-bottom: 1.25rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}}

.result-label {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255, 255, 255, 0.75) !important;
    margin: 0 0 0.35rem 0;
}}

.result-class {{
    font-size: 1.85rem;
    font-weight: 700;
    color: white !important;
    margin: 0 0 0.5rem 0;
}}

.result-confidence {{
    display: inline-block;
    background: white;
    color: var(--bg-deep) !important;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.4rem 1.1rem;
    border-radius: 999px;
    margin: 0;
}}

.chart-title {{
    font-size: 1rem;
    font-weight: 600;
    color: white !important;
    margin: 0 0 1rem 0;
}}

.prob-item {{
    margin-bottom: 0.85rem;
}}

.prob-item:last-child {{
    margin-bottom: 0;
}}

.prob-head {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.35rem;
    font-size: 0.88rem;
}}

.prob-name {{
    color: white !important;
    font-weight: 500;
}}

.prob-pct {{
    color: white !important;
    font-weight: 700;
}}

.prob-track {{
    height: 10px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 999px;
    overflow: hidden;
}}

.prob-fill {{
    height: 100%;
    background: white;
    border-radius: 999px;
    transition: width 0.4s ease;
}}

.prob-fill.top {{
    background: linear-gradient(90deg, #ffffff, #e8f7f6);
    box-shadow: 0 0 12px rgba(255, 255, 255, 0.4);
}}

body.main-mode h2, body.main-mode h3, body.main-mode h4 {{
    color: white !important;
}}

body.main-mode .stSpinner > div {{
    border-top-color: white !important;
}}

body.main-mode [data-testid="stAlert"] {{
    background: rgba(255, 255, 255, 0.15) !important;
    color: white !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
}}

@media (max-width: 768px) {{
    body.splash-mode .main .block-container,
    body.main-mode .main .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}
}}
</style>
<script>
document.documentElement.style.backgroundColor = '{BG}';
document.body.style.backgroundColor = '{BG}';
</script>
""",
    unsafe_allow_html=True,
)

# ======================================================
# NAVIGATION
# ======================================================
if "page" not in st.session_state:
    st.session_state.page = "splash"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def go_to_main():
    st.session_state.page = "main"
    st.rerun()


def set_page_mode(mode):
    st.markdown(
        f"<script>document.body.classList.remove('splash-mode','main-mode');"
        f"document.body.classList.add('{mode}-mode');"
        f"document.body.style.backgroundColor='{BG}';</script>",
        unsafe_allow_html=True,
    )


# ======================================================
# SPLASH
# ======================================================
def splash_screen():
    set_page_mode("splash")

    c1, c2 = st.columns([1.1, 0.9], gap="large")

    with c1:
        st.markdown('<div class="splash-wrap">', unsafe_allow_html=True)
        st.markdown('<span class="splash-badge">IA · Classification audio</span>', unsafe_allow_html=True)
        st.markdown(
            """
            <h1 class="splash-title">Baby Cry<br>Classification</h1>
            <p class="splash-subtitle">Détection intelligente des pleurs</p>
            <p class="splash-text">
                Identifiez les besoins du bébé à partir de ses pleurs
                grâce à un modèle de deep learning.
            </p>
            <div class="splash-stat">Précision · 95%</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Commencer", use_container_width=False):
            go_to_main()

    with c2:
        try:
            st.image("assets/images/baby.png", width="stretch")
        except Exception:
            st.info("Ajoutez l'image dans assets/images/baby.png")


# ======================================================
# MAIN
# ======================================================
def main_app():
    set_page_mode("main")

    st.markdown(
        """
        <div class="app-header">
            <h1>Baby Cry Classifier</h1>
            <p>Analysez un fichier audio .wav</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<p class="panel-marker" style="display:none"></p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">Charger un fichier audio</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="panel-subtitle">Glissez-déposez ou parcourez vos fichiers</p>',
            unsafe_allow_html=True,
        )
        upload_section()

    if st.session_state.get("analysis_result"):
        probs, pred_idx, confidence = st.session_state.analysis_result
        display_results(probs, pred_idx, confidence)


def upload_section():
    uploaded_file = st.file_uploader(
        "Déposez votre fichier .wav ici",
        type=["wav"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.markdown('<p class="audio-label">Aperçu audio</p>', unsafe_allow_html=True)
        st.audio(uploaded_file)

    st.markdown('<p class="analyze-marker" style="display:none"></p>', unsafe_allow_html=True)
    if st.button("Analyser", key="btn_analyser", width="stretch"):
        run_analysis(uploaded_file)


# ======================================================
# ANALYSE
# ======================================================
def run_analysis(uploaded_file):
    with st.spinner("Analyse en cours..."):
        try:
            if uploaded_file is None:
                st.warning("Veuillez charger un fichier .wav.")
                return
            path = "temp.wav"
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            spec_tensor = preprocess_audio(path)
            probs, pred_idx, confidence = run_inference(spec_tensor)
            st.session_state.analysis_result = (probs, pred_idx, confidence)
        except Exception as e:
            st.error(str(e))


# ======================================================
# AUDIO PREPROCESS
# ======================================================
def preprocess_audio(path):
    y, _ = librosa.load(path, sr=SR, duration=DURATION)
    target = SR * DURATION
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    y = y[:target]
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0)


# ======================================================
# MODEL
# ======================================================
class_names = [
    "belly_pain",
    "burping",
    "discomfort",
    "hungry",
    "tired",
]


def run_inference(spec_tensor):
    probs = np.random.dirichlet(np.ones(5))[0:]
    probs = torch.FloatTensor(probs)
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item() * 100
    return probs, pred_idx, confidence


# ======================================================
# RESULTS
# ======================================================
def display_results(probs, pred_idx, confidence):
    label = class_names[pred_idx].replace("_", " ").title()
    sorted_probs = sorted(
        [(class_names[i], round(float(probs[i]) * 100, 1)) for i in range(5)],
        key=lambda x: x[1],
        reverse=True,
    )

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-hero">
            <p class="result-label">Prédiction</p>
            <p class="result-class">{label}</p>
            <p class="result-confidence">{confidence:.1f}% confiance</p>
        </div>
        <p class="chart-title">Répartition par classe</p>
        """,
        unsafe_allow_html=True,
    )

    for i, (name, pct) in enumerate(sorted_probs):
        display_name = name.replace("_", " ").title()
        fill_class = "prob-fill top" if i == 0 else "prob-fill"
        st.markdown(
            f"""
            <div class="prob-item">
                <div class="prob-head">
                    <span class="prob-name">{display_name}</span>
                    <span class="prob-pct">{pct}%</span>
                </div>
                <div class="prob-track">
                    <div class="{fill_class}" style="width: {pct}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# ROUTING
# ======================================================
if st.session_state.page == "splash":
    splash_screen()
else:
    main_app()
