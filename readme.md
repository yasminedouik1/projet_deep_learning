# Baby Cry Classification

Classification des pleurs de bébé en 5 classes  
(belly_pain, burping, discomfort, hungry, tired)

---

## Description du Projet

Ce projet de Deep Learning a pour objectif de classifier automatiquement les pleurs de bébé selon leur cause.  
Développé dans le cadre d'un projet académique, il combine traitement du signal audio, augmentation de données et comparaison de plusieurs architectures de réseaux de neurones.

---

## Fonctionnalités

- Prétraitement audio et extraction de Mel-Spectrogrammes
- Augmentation de données pour gérer le déséquilibre des classes
- Comparaison de trois architectures : CNN, ResNet custom et CRNN
- Interprétabilité via Grad-CAM et courbes ROC
- Interface web interactive avec Streamlit (bonus)

---

## Dataset

- Source : Donateacry Corpus
- Nombre de classes : 5
- Nombre d'enregistrements : environ 1128
- Défi principal : fort déséquilibre entre les classes
- Solution : augmentation de données (Time Stretching, Pitch Shifting, Noise, SpecAugment)

---

## Pipeline Deep Learning

### 1. Prétraitement
- Chargement des fichiers WAV
- Resampling à 22.05 kHz
- Extraction de Mel-Spectrogrammes (128 mel bins)

### 2. Augmentation de Données
- Time stretching
- Pitch shifting
- Ajout de bruit gaussien
- SpecAugment (masquage temporel et fréquentiel)

### 3. Architectures Testées

| Modèle                | Type                  | Nombre de paramètres | Accuracy Test | Commentaire |
|-----------------------|-----------------------|----------------------|---------------|-----------|
| ImprovedSimpleCNN     | CNN amélioré          | ~1.3M                | 90.24%        | Meilleur modèle |
| BabyCryResNet         | ResNet custom         | ~2.8M                | 89.20%        | Réseau profond |
| CRNN                  | CNN + GRU             | ~1.1M                | 85.02%        | Modèle temporel |

**Modèle retenu** : ImprovedSimpleCNN

---

## Résultats

- Meilleure Accuracy sur Test Set : 90.24%
- F1-Macro : 90.18%
- Techniques utilisées pour limiter le surapprentissage : Early Stopping, Label Smoothing, Dropout, Batch Normalization, Weight Decay

---

## Installation et Utilisation

### 1. Cloner le dépôt
```bash
git clone https://github.com/votreusername/baby-cry-classification.git
cd baby-cry-classification

2. Installer les dépendances
Bashpip install -r requirements.txt
3. Lancer l'interface Streamlit
Bashstreamlit run app.py
4. Exécuter les notebooks

notebook1.ipynb : Prétraitement et augmentation
Baby_Cry_Classification_Final.ipynb : Version finale propre


Structure du Projet
textbaby-cry-classification/
├── notebooks/
│   ├── notebook1.ipynb
│   └── Baby_Cry_Classification_Final.ipynb
├── models.py
├── app.py
├── saved_models/
├── saved_data/
├── requirements.txt
├── README.md
└── presentation.pptx

Technologies Utilisées

PyTorch + TorchAudio
Librosa
Streamlit
Scikit-learn, Matplotlib, Seaborn


Perspectives d’Amélioration

Utilisation d’Audio Transformers (AST)
Déploiement sur Hugging Face Spaces
Développement d’une application mobile
Collecte de données supplémentaires


Auteur
Yasmine
Projet réalisé dans le cadre du cours de Deep Learning