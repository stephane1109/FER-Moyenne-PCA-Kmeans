##########################################
# Projet : FER - Moyenne - PCA - Kmeans
# Auteur : Stéphane Meurisse
# Contact : stephane.meurisse@example.com
# Site Web : https://www.codeandcortex.fr
# LinkedIn : https://www.linkedin.com/in/st%C3%A9phane-meurisse-27339055/
# Date : 18 octobre 2024
##########################################

# pip install opencv-python-headless fer pandas matplotlib altair xlsxwriter scikit-learn numpy streamlit tensorflow yt_dlp seaborn
# pip install tensorflow-metal -> pour Mac M2
# pip install vl-convert-python
# FFmpeg -> attention sous Mac la procédure d'installation sous MAC nécessite "Homebrew"

import streamlit as st
import subprocess
import os
import numpy as np
import seaborn as sns
from fer import FER
import cv2
from yt_dlp import YoutubeDL
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import shutil #suppression du repertoire
# import pysrt  # Nécessaire pour manipuler les fichiers SRT

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
    repertoire = repertoire.strip()
    repertoire = os.path.abspath(repertoire)

    # Si le répertoire existe déjà, suppression du contenu du répertoire des images
    images_25fps = os.path.join(repertoire, "images_25fps")
    if os.path.exists(images_25fps):
        st.write("Le répertoire des images existe déjà, suppression de tout le contenu en cours...")
        shutil.rmtree(images_25fps)  # Supprime tout le contenu du répertoire (fichiers et sous-dossiers)
        st.write(f"Le répertoire {images_25fps} et son contenu ont été supprimés.")

    # Création du répertoire de travail si nécessaire
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)
        st.write(f"Le répertoire a été créé : {repertoire}")
    else:
        st.write(f"Le répertoire existe déjà : {repertoire}")

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

###
# Fonction d'analyse d'émotion d'une image
def analyser_image(image_path, detector):
    if image_path is None:
        st.write(f"Aucune image extraite pour le chemin : {image_path}")
        return {}
    image = cv2.imread(image_path)
    if image is None:
        st.write(f"Impossible de lire l'image : {image_path}")
        return {}
    resultats = detector.detect_emotions(image)
    if resultats:
        for result in resultats:
            (x, y, w, h) = result["box"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            emotions = result['emotions']
            for idx, (emotion, score) in enumerate(emotions.items()):
                text = f"{emotion}: {score:.4f}"
                cv2.putText(image, text, (x, y + h + 20 + (idx * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(image_path, image)
        return resultats[0]['emotions']
    else:
        st.write(f"Aucune émotion détectée dans l'image {image_path}")
        return {}
###



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

### Fonction pour analyse de la variance
# Fonction pour calculer la moyenne et la variance des émotions
def moyenne_et_variance_par_emotion(emotions_list):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    resultats = {}
    for emotion in emotions:
        emotion_scores = [emotion_dict.get(emotion, 0) for emotion_dict in emotions_list]
        moyenne = np.mean(emotion_scores)
        variance = np.var(emotion_scores)
        resultats[emotion] = {'moyenne': moyenne, 'variance': variance}
    return resultats
###

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

    ###
    # Ajout du streamgraph pour les émotions par frame
    df_emotions['Frame_Index'] = df_emotions.apply(lambda x: x['Seconde'] * 25 + int(x['Frame'].split('_')[1]), axis=1)
    df_streamgraph_frames = df_emotions.melt(id_vars=['Frame_Index', 'Seconde'],
                                             value_vars=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise',
                                                         'neutral'],
                                             var_name='Emotion',
                                             value_name='Score')

    streamgraph_frames = alt.Chart(df_streamgraph_frames).mark_area().encode(
        x=alt.X('Frame_Index:Q', title='Frame Index'),
        y=alt.Y('Score:Q', title='Score des émotions', stack='center'),
        color=alt.Color('Emotion:N', title='Émotion'),
        tooltip=['Frame_Index', 'Emotion', 'Score']
    ).properties(
        title='Streamgraph des émotions par frame (25 fps)',
        width=800,
        height=400
    )

    st.write("#### Streamgraph des émotions par frame (25fps)")
    st.altair_chart(streamgraph_frames, use_container_width=True)
    ###

    st.write("Moyenne des émotions par seconde")
    st.dataframe(df_emotion_dominante_moyenne)

    ###
    # Ajout du streamgraph pour les moyennes des émotions par seconde
    df_streamgraph_seconds = df_emotion_dominante_moyenne.melt(
        id_vars=['Seconde'],
        value_vars=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        var_name='Emotion',
        value_name='Score'
    )

    streamgraph_seconds = alt.Chart(df_streamgraph_seconds).mark_area().encode(
        x=alt.X('Seconde:Q', title=f'Secondes (de {start_time} à {end_time})'),
        y=alt.Y('Score:Q', title='Score des émotions', stack='center'),
        color=alt.Color('Emotion:N', title='Émotion'),
        tooltip=['Seconde', 'Emotion', 'Score']
    ).properties(
        title='Streamgraph des moyennes des émotions par seconde',
        width=800,
        height=400
    )

    st.write("#### Streamgraph des moyennes des émotions par seconde")
    st.altair_chart(streamgraph_seconds, use_container_width=True)
####

####
    # Ajout du calcul de la variance et de la moyenne par seconde

    # Calcul des moyennes et variances des émotions par seconde
    stats_par_seconde = moyenne_et_variance_par_emotion(emotion_dominante_moyenne_results)

    if stats_par_seconde:
        # Convertir les résultats en DataFrame pour affichage
        df_stats_seconde = pd.DataFrame(stats_par_seconde).T.reset_index()
        df_stats_seconde.columns = ['Emotion', 'Moyenne', 'Variance']

        # Afficher la DataFrame des moyennes et variances
        st.write("#### Tableau des moyennes et variances des émotions par seconde")
        st.dataframe(df_stats_seconde)

        # Création du graphique combinant Moyenne et Variance
        st.write("#### Graphique des moyennes et variances des émotions par seconde")

        # Barres pour les moyennes
        moyenne_bar_seconde = alt.Chart(df_stats_seconde).mark_bar().encode(
            x=alt.X('Emotion:N', title='Émotion'),
            y=alt.Y('Moyenne:Q', title='Moyenne des probabilités'),
            color=alt.Color('Emotion:N', legend=None)
        )

        # Points pour les variances
        variance_point_seconde = alt.Chart(df_stats_seconde).mark_circle(size=100, color='red').encode(
            x=alt.X('Emotion:N', title='Émotion'),
            y=alt.Y('Variance:Q', title='Variance des probabilités'),
            tooltip=['Emotion', 'Variance']
        )

        # Superposer les deux graphiques
        graphique_combine_seconde = alt.layer(moyenne_bar_seconde, variance_point_seconde).resolve_scale(
            y='independent'
        ).properties(
            width=600,
            height=400,
        )

        # Affichage du graphique
        st.altair_chart(graphique_combine_seconde, use_container_width=True)
    else:
        st.write("Aucune donnée disponible pour les moyennes et variances.")
#####

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

###
    # Calcul des moyennes des émotions par cluster
    cluster_means = df_emotion_dominante_moyenne.groupby('Cluster')[emotions].mean()

    # 1. Calcul de la similarité cosinus entre les centroïdes des clusters
    centroids_similarity = cosine_similarity(centroids)  # Obtenez les centroïdes de KMeans
    df_similarity_centroids = pd.DataFrame(centroids_similarity,
                                           index=[f'Cluster {i}' for i in range(n_clusters)],
                                           columns=[f'Cluster {i}' for i in range(n_clusters)])

    # 2. Calcul de la similarité cosinus entre les moyennes des émotions par cluster
    emotion_similarity = cosine_similarity(cluster_means.values)
    df_similarity_means = pd.DataFrame(emotion_similarity,
                                       index=[f'Cluster {i}' for i in range(n_clusters)],
                                       columns=[f'Cluster {i}' for i in range(n_clusters)])

    # 3. Affichage des similarités cosinus entre les centroïdes dans Streamlit
    st.subheader("Similarité cosinus entre les centroïdes des clusters")
    st.write("Les valeurs proches de 1 indiquent que les clusters sont très similaires.")
    st.dataframe(df_similarity_centroids)

    # 4. Affichage des similarités cosinus entre les moyennes des émotions dans Streamlit
    st.subheader("Similarité cosinus entre les moyennes des émotions par cluster")
    st.write("Les valeurs proches de 1 indiquent que les moyennes des émotions entre clusters sont très similaires.")
    st.dataframe(df_similarity_means)

    # 5. Heatmap pour la similarité cosinus entre les centroïdes des clusters
    st.subheader("Heatmap - Similarité cosinus entre les centroïdes des clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_similarity_centroids, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # 6. Heatmap pour la similarité cosinus entre les moyennes des émotions dans chaque cluster
    st.subheader("Heatmap - Similarité cosinus entre les moyennes des émotions dans chaque cluster")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_similarity_means, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
###

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

# Explication des résultats en Markdown
st.markdown("""
### Interprétation des résultats de similarité cosinus

- **Similarité cosinus entre les centroïdes des clusters** : 
    - Cette mesure vous permet de comprendre à quel point les clusters sont proches les uns des autres dans l'espace des composantes principales (PCA).
    - Une valeur proche de 1 indique que deux clusters sont très similaires dans cet espace. Cela peut signifier que ces clusters capturent des combinaisons émotionnelles proches dans l'analyse.

- **Similarité cosinus entre les moyennes des émotions par cluster** :
    - Cette mesure compare les **moyennes des émotions** de chaque cluster.
    - Une valeur proche de 1 dans cette matrice indique que deux clusters partagent un **profil émotionnel** similaire, même si KMeans les a séparés en deux groupes.
    - Si des clusters ont des similarités cosinus élevées, il peut être intéressant de les regrouper pour une analyse plus fine, car cela peut indiquer qu'ils capturent des tendances émotionnelles semblables.
""")

