##########################################
# Projet : FER - Moyenne - PCA - Kmeans
# Auteur : Stéphane Meurisse
# Contact : stephane.meurisse@example.com
# Site Web : https://www.codeandcortex.fr
# LinkedIn : https://www.linkedin.com/in/st%C3%A9phane-meurisse-27339055/
# Date : 17 octobre 2024
##########################################

# pip install opencv-python-headless fer pandas matplotlib altair xlsxwriter scikit-learn numpy streamlit tensorflow yt_dlp seaborn
# pip install tensorflow-metal -> pour Mac M2
# pip install vl-convert-python
# FFmpeg -> attention sous Mac la procédure d'installation sous MAC nécessite "Homebrew"

import streamlit as st
import subprocess
import os
import pandas as pd
import numpy as np
from collections import Counter
from fer import FER
import cv2
from yt_dlp import YoutubeDL
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# Fonction pour vider le cache
def vider_cache():
    st.cache_resource.clear()
    st.write("Cache vidé systématiquement au lancement du script")


# Fonction pour définir le répertoire de travail
def definir_repertoire_travail():
    repertoire = st.text_input("Définir le répertoire de travail", "", key="repertoire_travail")
    if not repertoire:
        st.write("Veuillez spécifier un chemin valide.")
        return ""
    repertoire = os.path.abspath(repertoire.strip())
    os.makedirs(repertoire, exist_ok=True)
    st.write(f"Répertoire de travail : {repertoire}")
    return repertoire


# Fonction pour télécharger la vidéo avec ytdlp
def telecharger_video(url, repertoire):
    video_path = os.path.join(repertoire, 'video.mp4')
    if os.path.exists(video_path):
        st.write(f"La vidéo est déjà présente : {video_path}")
        return video_path
    st.write(f"Téléchargement de la vidéo depuis {url}...")
    ydl_opts = {'outtmpl': video_path, 'format': 'best'}
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    st.write(f"Téléchargement terminé : {video_path}")
    return video_path


# Fonction pour extraire des images à 25fps avec FFmpeg
def extraire_images_25fps_ffmpeg(video_path, repertoire, seconde):
    images_extraites = []
    for frame in range(25):
        image_path = os.path.join(repertoire, f"image_25fps_{seconde}_{frame}.jpg")
        if os.path.exists(image_path):
            images_extraites.append(image_path)
            continue
        time = seconde + frame * (1 / 25)
        cmd = ['ffmpeg', '-ss', str(time), '-i', video_path, '-frames:v', '1', '-q:v', '2', image_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            st.write(f"Erreur FFmpeg à {time} seconde : {result.stderr.decode('utf-8')}")
            break
        images_extraites.append(image_path)
    return images_extraites


# Fonction d'analyse d'émotion d'une image
def analyser_image(image_path, detector):
    if image_path is None:
        return {}
    image = cv2.imread(image_path)
    if image is None:
        return {}
    resultats = detector.detect_emotions(image)
    return resultats[0]['emotions'] if resultats else {}


# Calcul de l'émotion dominante par moyenne des scores
def emotion_dominante_par_moyenne(emotions_list):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    moyenne_emotions = {emotion: np.mean([emo.get(emotion, 0) for emo in emotions_list]) for emotion in emotions}
    emotion_dominante = max(moyenne_emotions, key=moyenne_emotions.get)
    return moyenne_emotions, emotion_dominante


# Fonction pour optimiser le nombre de clusters
def optimiser_clusters(X_pca):
    scores = []
    range_n_clusters = list(range(2, 10))  # Tester pour des clusters de 2 à 10

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_pca)
        score = silhouette_score(X_pca, kmeans.labels_)
        scores.append(score)

    # Création du DataFrame pour Altair
    df_silhouette = pd.DataFrame({
        'Nombre de clusters': range_n_clusters,
        'Score de silhouette': scores
    })

    # Création du graphique avec Altair
    chart = alt.Chart(df_silhouette).mark_line(point=True).encode(
        x='Nombre de clusters:Q',
        y='Score de silhouette:Q',
        tooltip=['Nombre de clusters', 'Score de silhouette']
    ).properties(
        title="Score de silhouette en fonction du nombre de clusters",
        width=600,
        height=400
    )

    # Affichage du graphique dans Streamlit
    st.altair_chart(chart, use_container_width=True)

    # Retourne le nombre de clusters avec le meilleur score de silhouette
    return range_n_clusters[scores.index(max(scores))]


# Fonction principale pour analyser la vidéo
def analyser_video(video_url, start_time, end_time, repertoire_travail):
    st.write(f"Analyse de la vidéo entre {start_time} et {end_time} seconde(s)")
    repertoire_25fps = os.path.join(repertoire_travail, "images_25fps")
    os.makedirs(repertoire_25fps, exist_ok=True)
    video_path = telecharger_video(video_url, repertoire_travail)
    detector = FER()

    results_25fps = []
    emotion_dominante_moyenne_results = []

    # Analyse de chaque seconde de la vidéo
    for seconde in range(start_time, end_time + 1):
        images_25fps = extraire_images_25fps_ffmpeg(video_path, repertoire_25fps, seconde)
        emotions_25fps_list = [analyser_image(image_path, detector) for image_path in images_25fps]

        # Stockage des résultats pour chaque frame
        for idx, emotions in enumerate(emotions_25fps_list):
            results_25fps.append({'Seconde': seconde, 'Frame': f'25fps_{seconde * 25 + idx}', **emotions})

        # Calcul de la moyenne des émotions pour la seconde
        moyenne_emotions, _ = emotion_dominante_par_moyenne(emotions_25fps_list)
        emotion_dominante_moyenne_results.append({'Seconde': seconde, **moyenne_emotions})

    # Création des DataFrames
    df_emotions = pd.DataFrame(results_25fps)
    df_emotion_dominante_moyenne = pd.DataFrame(emotion_dominante_moyenne_results)

    # Affichage des DataFrames originales
    st.subheader("Données originales")
    st.write("Scores des émotions par frame (25 fps)")
    st.dataframe(df_emotions)

    st.write("Moyenne des émotions par seconde")
    st.dataframe(df_emotion_dominante_moyenne)

    # Préparation des données pour le clustering et PCA
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    X = df_emotion_dominante_moyenne[emotions].values

    # Normalisation et PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Application de la PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Extraction de la variance expliquée par chaque composante principale
    explained_variance_ratio = pca.explained_variance_ratio_
    df_variance_expliquee = pd.DataFrame({
        'Composante Principale': [f'PC{i + 1}' for i in range(len(explained_variance_ratio))],
        'Variance expliquée (%)': explained_variance_ratio * 100
    })

    # Affichage du tableau des variances expliquées dans Streamlit
    st.subheader("Variance expliquée par les composantes principales")
    st.write(df_variance_expliquee)

    # Affichage d'un graphique des variances expliquées avec Altair
    chart_variance = alt.Chart(df_variance_expliquee).mark_bar().encode(
        x=alt.X('Composante Principale', title='Composante Principale'),
        y=alt.Y('Variance expliquée (%)', title='Variance expliquée (%)', scale=alt.Scale(domain=[0, 100])),
        tooltip=['Composante Principale', 'Variance expliquée (%)']
    ).properties(
        title="Variance expliquée par chaque composante principale",
        width=600,
        height=400
    )
    st.altair_chart(chart_variance, use_container_width=True)

    # On garde les deux premières composantes pour l'affichage PCA
    df_emotion_dominante_moyenne['PC1'] = X_pca[:, 0]
    df_emotion_dominante_moyenne['PC2'] = X_pca[:, 1]

    # Optimisation du nombre de clusters et affichage de la courbe du score de silhouette
    n_clusters = optimiser_clusters(X_pca)
    st.write(f"Nombre optimal de clusters : {n_clusters}")

    # Application de K-means sur les données PCA
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_emotion_dominante_moyenne['Cluster'] = kmeans.fit_predict(X_pca)

    # Création du graphique de clustering avec PC1 et PC2
    st.subheader("Visualisation du clustering (PCA)")
    cluster_chart = alt.Chart(df_emotion_dominante_moyenne).mark_circle(size=60).encode(
        x=alt.X('PC1:Q', title='Première Composante Principale'),
        y=alt.Y('PC2:Q', title='Deuxième Composante Principale'),
        color=alt.Color('Cluster:N', scale=alt.Scale(scheme='category10')),
        tooltip=['Seconde', 'Cluster'] + emotions
    ).properties(
        width=600,
        height=400,
        title='Clustering K-means des émotions (PCA)'
    )

    # Ajout des centroïdes
    centroids = kmeans.cluster_centers_

    # Limite les centroïdes aux deux premières composantes (PC1, PC2)
    df_centroids = pd.DataFrame(centroids[:, :2], columns=['PC1', 'PC2'])
    df_centroids['Cluster'] = range(n_clusters)

    # Création du graphique des centroïdes
    centroid_chart = alt.Chart(df_centroids).mark_point(size=200, shape='cross', filled=True).encode(
        x='PC1:Q',
        y='PC2:Q',
        color=alt.Color('Cluster:N', scale=alt.Scale(scheme='category10'))
    )

    # Affichage du clustering avec les centroïdes
    clustering_plot = cluster_chart + centroid_chart
    st.altair_chart(clustering_plot, use_container_width=True)

    # Évolution des clusters au fil du temps
    timeline_chart = alt.Chart(df_emotion_dominante_moyenne).mark_rect().encode(
        x=alt.X('Seconde:O', title='Seconde'),
        y=alt.Y('Cluster:N', title='Cluster'),
        color=alt.Color('Cluster:N', scale=alt.Scale(scheme='category10'))
    ).properties(
        width=800,
        height=200,
        title="Évolution des clusters au fil du temps"
    )
    st.altair_chart(timeline_chart, use_container_width=True)

    # Analyse des caractéristiques de chaque cluster
    st.subheader("Caractéristiques des clusters")
    for cluster in range(n_clusters):
        cluster_data = df_emotion_dominante_moyenne[df_emotion_dominante_moyenne['Cluster'] == cluster]
        st.write(f"Cluster {cluster}:")
        cluster_means = cluster_data[emotions].mean().sort_values(ascending=False)
        st.write("Moyennes des scores émotionnels:")
        st.write(cluster_means)
        st.write("---")


# Interface Streamlit
st.title("Analyse des émotions avec clustering K-means")
st.markdown("<h6 style='text-align: center;'>www.codeandcortex.fr</h5>", unsafe_allow_html=True)

vider_cache()
repertoire_travail = definir_repertoire_travail()

video_url = st.text_input("URL de la vidéo à analyser", "", key="video_url")
start_time = st.number_input("Temps de départ de l'analyse (en secondes)", min_value=0, value=0, key="start_time")
end_time = st.number_input("Temps d'arrivée de l'analyse (en secondes)", min_value=start_time, value=start_time + 1,
                           key="end_time")

if st.button("Lancer l'analyse"):
    if video_url and repertoire_travail:
        analyser_video(video_url, start_time, end_time, repertoire_travail)
    else:
        st.write("Veuillez définir le répertoire de travail et l'URL de la vidéo.")






