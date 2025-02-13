import streamlit as st
import torch
import cv2
import numpy as np
import time
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
import tempfile
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

# Initialisation du modèle
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Vérifier si la liste des fichiers audio existe
if "audio_files" not in st.session_state:
    st.session_state.audio_files = []

# Fonction de traduction
def translate_text(text, src="en", dest="fr"):
    try:
        url = f"https://api.mymemory.translated.net/get?q={text}&langpair={src}|{dest}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("responseData", {}).get("translatedText", "Erreur de traduction")
    except Exception as e:
        return f"Erreur de connexion : {e}"

# Fonction pour décrire la vidéo 
# Fonction pour décrire une image
def describe_image(image):
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    description_en = processor.decode(output[0], skip_special_tokens=True)
    
    description_fr = translate_text(description_en, src="en", dest="fr")
    description_wo = translate_text(description_en, src="en", dest="wo")
    
    return description_en, description_fr, description_wo

# Fonction pour générer l'audio
def text_to_speech(text, lang):
    try:
        tts = gTTS(text, lang=lang)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        st.session_state.audio_files.append(temp_file.name)
        return temp_file.name
    except Exception as e:
        return None

# Fonction pour supprimer un fichier audio
def delete_audio(file):
    if file in st.session_state.audio_files:
        os.remove(file)
        st.session_state.audio_files.remove(file)
        st.rerun()  # Rafraîchir la page

# Interface Streamlit
st.title("🎥 Vision AI : Analyse Vidéo en Direct")

# Afficher les fichiers audio avec bouton de suppression
st.subheader("🎵 Audios générés :")

# Utilisation de st.container() pour éviter le problème de suppression en boucle
with st.container():
    for idx, audio_file in enumerate(st.session_state.audio_files):
        col1, col2 = st.columns([5, 1])  # Ajustement des colonnes

        with col1:
            st.audio(audio_file, format="audio/mp3")

        with col2:
            if st.button(key=f"delete_{idx}"):  # Identifiant unique
                delete_audio(audio_file)

# Activer la capture vidéo
run = st.checkbox("📹 Démarrer la vidéo")

# Sélectionner la langue
lang_choice = st.selectbox("🗣️ Choisissez la langue :", ["🇬🇧 Anglais", "🇫🇷 Français", "🇸🇳 Wolof"])

# Charger une police compatible avec les accents
try:
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(font_path, 40)
except:
    font = ImageFont.load_default()

# Ouvrir la caméra
if run:
    cap = cv2.VideoCapture(0)  # 0 = Webcam par défaut
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur lors de la capture vidéo")
            break
        
        # Convertir la frame en image PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Générer la description
        description_en, description_fr, description_wo = describe_image(img_pil)

        # Sélectionner la description selon la langue
        if lang_choice == "🇬🇧 Anglais":
            description = description_en
            audio_lang = "en"
        elif lang_choice == "🇫🇷 Français":
            description = description_fr
            audio_lang = "fr"
        else:
            description = description_wo
            audio_lang = "wo"

        # Définir la position et la taille du fond derrière le texte
        text_x, text_y = 20, 50
        text_bbox = draw.textbbox((0, 0), description, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 20  # Espace autour du texte
        border_radius = 15  # Rayon des coins arrondis

        # Couleur bleu 
        bg_color = (173, 216, 230, 180)  # Bleu ciel semi-transparent

        # Créer une nouvelle image transparente
        overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Dessiner un rectangle arrondi en arrière-plan pour y mettre la description (ameliorer l'affichage de la description)
        x1, y1 = text_x - padding, text_y - padding
        x2, y2 = text_x + text_width + padding, text_y + text_height + padding

        overlay_draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=border_radius, fill=bg_color)

        # Fusionner l'overlay avec l'image originale
        img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay)

        # Ajouter la description sur l'image
        draw = ImageDraw.Draw(img_pil)
        draw.text((text_x, text_y), description, font=font, fill=(255, 255, 255))

        # Convertir l'image PIL en OpenCV
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Mettre à jour le flux vidéo
        stframe.image(frame, channels="BGR")

        # Générer et jouer l'audio
        audio_path = text_to_speech(description, audio_lang)
        if audio_path:
            st.audio(audio_path, format="audio/mp3")

        # Attendre 5 secondes avant la prochaine capture
        time.sleep(5)

    cap.release()
