import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from models import muq, RoBERTa
from eval import GridSearchEvaluator


SAMPLING_RATE = 44100

def load_soundfiles(path):
    soundfiles = []
    for file in path.iterdir():
        if file.is_dir():
            soundfiles = load_soundfiles(file)
        else:
            if file.suffix in [".wav", ".aif", ".mp3", ".m4a"]:
                data, sr = librosa.load(file, sr=SAMPLING_RATE)
                assert sr == SAMPLING_RATE
                if data.ndim > 1:  # convert to mono
                    data = data.sum(axis=1)
                    data = data / np.abs(data).max()
                soundfiles.append(data)
    return soundfiles


def remove_silence(sounds, top_db=60):
    """
    Removes leading and trailing silence from a list of sound arrays.

    Args:
        sounds (list of np.ndarray): List of audio signals as numpy arrays.
        top_db (int): Threshold (in decibels) below reference to consider as silence.

    Returns:
        list of np.ndarray: List of trimmed audio signals.
    """
    trimmed_sounds = []
    for sound in sounds:
        # Check if the whole sound is below the top_db threshold
        if librosa.get_duration(y=sound) == 0 or np.max(np.abs(sound)) < librosa.db_to_amplitude(-top_db):
            continue

        # Trim silence from the beginning and end
        trimmed_sound, _ = librosa.effects.trim(sound, top_db=top_db)
        trimmed_sounds.append(trimmed_sound)
    return trimmed_sounds


def preprocess_sounds(sounds, slice_fn, trim_silence=True):
    if trim_silence:
        sounds = remove_silence(sounds)
    processed_sounds = []
    for sound in sounds:
        slices = slice_fn(sound)
        for slice in slices:
            processed_sounds.append(slice)
    return processed_sounds


# what transformations will we apply to the feature spaces?
def create_pipeline(normalization_method, dim):
    pca = PCA(n_components=dim)
    return [normalization_method.fit_transform, pca.fit_transform]


def load_text_corpus(path):
    assert path.is_file()
    with open(path, "r") as f:
        return [word for line in f.readlines() for word in line.split()]


def embed_sounds(sounds, encoder):
    embeddings = []
    for sound in sounds:
        embedding = encoder(sound, SAMPLING_RATE).squeeze()
        # For MuQ: average across time
        embedding = torch.mean(embedding, dim=0).flatten()
        embeddings.append(embedding)
    return torch.stack(embeddings)


def embed_text(text, encoder):
    return encoder(text)


def find_nearest_neighbors(S, points, distance_metric):
    """

    :param S: The space S of all sound embeddings
    :param points: A list of points that you want to find the nearest neighbors in S
    :param distance_metric: 'euclidean' or 'cosine'
    :return: (the nearest point, the idx of the nearest point)
    """
    neigh = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=distance_metric).fit(S)
    neighbor_indices = neigh.kneighbors(points, return_distance=False).flatten()
    return S[neighbor_indices], neighbor_indices


def save_output(sound_list, output_path):
    # concatenate audio files
    output = np.concatenate(sound_list)

    output_path.mkdir(parents=True, exist_ok=True)
    counter = 0
    while (output_path / f"output{counter}.wav").exists():
        counter += 1

    filename = output_path / f"output{counter}.wav"
    sf.write(filename, output, SAMPLING_RATE)

def equal_slices(sound, grain_size):
    if sound.size < grain_size:
        return [sound]
    return librosa.util.frame(sound, frame_length=grain_size, hop_length=grain_size, axis=0)


def evaluate_clustering(X, labels):
    eval = {}

    silhouette_avg = silhouette_score(X, labels)  # higher is better. The best value is 1 and the worst value is -1
    ch_score = calinski_harabasz_score(X, labels) # higher is better
    db_score = davies_bouldin_score(X, labels)  # lower is better. The best value is 0.

    eval["silhouette_score (higher is better)"] = silhouette_avg
    eval["calinski_harabasz_score (higher is better)"] = ch_score
    eval["davies_bouldin_score (lower is better)"] = db_score

    return eval


#####################
# MAPPING FUNCTIONS #
#####################


def identity(sound_embeddings, text_embeddings, distance_metric):
    W = np.eye(sound_embeddings.shape[1])
    mapped_text_embeddings = [W @ emb for emb in text_embeddings]
    return mapped_text_embeddings


def cluster_map(sound_embeddings, text_embeddings, distance_metric):
    print("Applying clustering...")
    k = 2
    sound_kmeans = KMeans(n_clusters=k, n_init=10).fit(sound_embeddings)
    text_kmeans = KMeans(n_clusters=k, n_init=10).fit(text_embeddings)

    sound_cluster_centers = sound_kmeans.cluster_centers_
    text_cluster_centers = text_kmeans.cluster_centers_

    sound_cluster_labels = sound_kmeans.labels_
    text_cluster_labels = text_kmeans.labels_

    # evaluate clustering
    sound_eval = evaluate_clustering(sound_embeddings, sound_cluster_labels)
    text_eval = evaluate_clustering(text_embeddings, text_cluster_labels)

    for k, v in sound_eval.items():
        print(f"Sound {k}: {v}")
    print("\n")
    for k, v in text_eval.items():
        print(f"Text {k}: {v}")

    # for each text cluster, find the nearest sound cluster
    nearest_cluster = {}  # maps a text cluster idx to the idx of the nearest sound cluster
    for i in range(k):
        center = text_cluster_centers[i]
        _, nearest_sound_cluster_idx = find_nearest_neighbors(sound_cluster_centers, [center], distance_metric)
        nearest_cluster[i] = nearest_sound_cluster_idx

    nearest_sounds = []
    # for each text embedding, find the nearest sound embedding that is in the nearest cluster
    for i, emb in enumerate(text_embeddings):
        text_cluster = text_kmeans.labels_[i]
        nearest_sound_cluster = nearest_cluster[text_cluster]  # idx of nearest sound cluster

        # get a list of the sound emebeddings in that cluster (this should be done in a separate loop)
        cluster_sound_idx = np.where(sound_kmeans.labels_ == nearest_sound_cluster)[0]  # list idx of sounds in that cluster
        cluster_sounds = sound_embeddings[cluster_sound_idx]

        # find the sound in this cluster that is nearest the text embedding
        nn, _ = find_nearest_neighbors(cluster_sounds, [emb], distance_metric)  # emb of nearest sound
        nearest_sounds.append(nn)

    # convert each sound embedding to the index it corresponds to in sound_corpus
    nearest_sound_idx = [np.where((sound_embeddings == sound))[0][0] for sound in nearest_sounds]
    return nearest_sound_idx


def run(params):

    normalization = StandardScaler()
    dim = 2  # the number of dimensions to reduce to

    # a function that determines how to separate sounds
    if params['grain_size'] is not None:
        grain_size = int(params['grain_size'] / 1000.0 * SAMPLING_RATE)  # convert to samples
        slice_fn = lambda y: equal_slices(y, grain_size)
    else:
        slice_fn = lambda y: y

    sound_corpus_path = Path(params["sound_corpus_path"])
    text_corpus_path = Path(params["text_corpus_path"])

    if params["sound_encoder"] == "MuQ":
        sound_encoder = muq
    if params["text_encoder"] == "RoBERTa":
        text_encoder = RoBERTa

    mapping = params["mapping"]

    print("Loading sound and text data...")
    sound_corpus = load_soundfiles(sound_corpus_path)
    sound_corpus = preprocess_sounds(sound_corpus, slice_fn, params["trim_silence"])

    text_corpus = load_text_corpus(text_corpus_path)

    print("Embedding sounds...")
    sound_embeddings = embed_sounds(sound_corpus, sound_encoder)

    print("Embedding text...")
    text_embeddings = embed_text(" ".join(text_corpus), text_encoder)

    print("Transforming embeddings...")
    # what transformations will we apply to the feature space?
    transform_pipeline = create_pipeline(normalization, dim)
    for transform in transform_pipeline:
        sound_embeddings = transform(sound_embeddings)
        text_embeddings = transform(text_embeddings)

    print(f"Mapping text to sound with method {mapping}...")
    # mapping options: identity, cluster_map
    if mapping == "identity":
        mapping_fn = identity
    elif mapping == "cluster":
        mapping_fn = cluster_map
    else:
        raise Exception("Invalid mapping provided")

    neighbor_indices = mapping_fn(sound_embeddings, text_embeddings, params['distance'])

    print("Fetching sounds...")
    output_sounds = [sound_corpus[i] for i in neighbor_indices]

    print("Saving output...")
    save_output(output_sounds, Path(params["output_path"]))

    print("Done.")

if __name__ == "__main__":

    parameters = {
        "sound_corpus_path": "./corpora/sound/toy",
        "text_corpus_path": "./corpora/text/test.txt",
        "sound_encoder": "MuQ",
        "text_encoder": "RoBERTa",
        "mapping": "cluster",
        "output_path": "./output",
        "grain_size": 3000,  # in ms
        "distance": "euclidean",
        "trim_silence": True,
    }

    run(parameters)

