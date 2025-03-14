import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
import torch
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from scipy.spatial import distance

from models import muq, RoBERTa, word2vec, fastText
from util import *


SAMPLING_RATE = 44100
OUTPUT_PATH = Path("./output")

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
    :return: list of (the nearest point, the idx of the nearest point)
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

#################
# SLICE METHODS #
#################

def entire_sample(sound):
    return sound


def equal_slices(sound, grain_size):
    if sound.size < grain_size:
        return [sound]
    return librosa.util.frame(sound, frame_length=grain_size, hop_length=grain_size, axis=0)


def get_onsets(y):
    # Detect onsets
    onset_samples = librosa.onset.onset_detect(y=y, sr=SAMPLING_RATE, units='samples')

    # Split audio at onsets
    segments = []
    for i in range(len(onset_samples) - 1):
        segment = y[onset_samples[i]:onset_samples[i + 1]]
        segments.append(segment)

    # Add final segment (after last onset to end)
    if onset_samples[-1] < len(y):
        segments.append(y[onset_samples[-1]:])

    return segments


def evaluate_clustering(X, labels):
    eval = {}

    silhouette_avg = silhouette_score(X, labels)  # higher is better. The best value is 1 and the worst value is -1
    ch_score = calinski_harabasz_score(X, labels) # higher is better
    db_score = davies_bouldin_score(X, labels)  # lower is better. The best value is 0.

    eval["silhouette_score (higher is better)"] = silhouette_avg
    eval["calinski_harabasz_score (higher is better)"] = ch_score
    eval["davies_bouldin_score (lower is better)"] = db_score

    return eval


def evaluate_mapping(t_embs, s_embs, distance_metric):
    """
    t_embs[i] is the sound mapped from t_embs[i]
    :param t_embs:
    :param s_embs:
    :return: a distance
    """
    distance = 0
    num_t_embs = t_embs.shape[0]
    pairs = list(combinations(range(num_t_embs), 2))  # list of all possible pairs of indices
    for i, j in pairs:
        t1, t2 = t_embs[i], t_embs[j]
        s1, s2 = s_embs[i], s_embs[j]
        dt = distance_metric(t1, t2)
        ds = distance_metric(s1, s2)
        diff = abs(dt - ds)
        distance += diff
    distance /= len(pairs)  # normalize by the number of pairs
    return distance


#####################
# MAPPING FUNCTIONS #
#####################


def identity(sound_embeddings, text_embeddings, distance_metric):
    W = np.eye(sound_embeddings.shape[1])
    mapped_text_embeddings = [W @ emb for emb in text_embeddings]
    _, nn_idx = find_nearest_neighbors(sound_embeddings, mapped_text_embeddings, distance_metric)
    return nn_idx


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

    for key, val in sound_eval.items():
        print(f"Sound {key}: {val}")
    print("\n")
    for key, val in text_eval.items():
        print(f"Text {key}: {val}")

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
    sound_corpus_path = Path(params.sound_path)
    text_corpus_path = Path(params.text_path)

    if params.sound_encoder == "MuQ":
        sound_encoder = muq

    if params.text_encoder == "RoBERTa":
        text_encoder = RoBERTa
    elif params.text_encoder == "word2vec":
        text_encoder = word2vec
    elif params.text_encoder == "fastText":
        text_encoder = fastText

    print("Loading sound and text data...")
    sound_corpus = load_soundfiles(sound_corpus_path)
    text_corpus = load_text_corpus(text_corpus_path)

    # check to see if this will be a valid run
    if params.dim > len(sound_corpus) or params.dim > len(text_corpus):
        print(f"!!!: Input PCA dimension {params.dim} cannot be larger than length of sound corpus ({len(sound_corpus)}) or text corpus ({len(text_corpus)})")
        print("Ending this run.")
        return

    # slice_fn: a function that determines how to separate sounds
    try:
        grain_size = int(params.sound_preprocessing / 1000.0 * SAMPLING_RATE)
        slice_fn = lambda y: equal_slices(y, grain_size)
    except TypeError:
        if params.sound_preprocessing == "onsets":
            slice_fn = lambda y: get_onsets(y)
        elif params.sound_preprocessing == "full":
            slice_fn = lambda y: y
        else:
            raise ValueError(f"Unknown preprocessing {params.sound_preprocessing}")
    sound_corpus = preprocess_sounds(sound_corpus, slice_fn, params.trim_silence)

    if params.sound_encoder == "MuQ":
        sound_corpus = [s for s in sound_corpus if s.size > 1024]  # MuQ requires sounds longer than 1024 samples

    print("Embedding sounds...")
    sound_embeddings = embed_sounds(sound_corpus, sound_encoder)

    print("Embedding text...")
    text_embeddings = embed_text(" ".join(text_corpus), text_encoder)

    # check again to see if this will be a valid run
    if params.dim > len(sound_embeddings) or params.dim > len(text_embeddings):
        print(
            f"!!!: Input PCA dimension {params.dim} cannot be larger than length of sound embeds ({len(sound_embeddings)}) or text embeds ({len(text_embeddings)})")
        print("Ending this run.")
        return

    print("Transforming embeddings...")
    # what transformations will we apply to the feature space?
    if params.normalization == "standard":
        norm_method = StandardScaler()
    else:
        raise KeyError("Unknown normalization method")

    transform_pipeline = create_pipeline(norm_method, params.dim)
    for transform in transform_pipeline:
        sound_embeddings = transform(sound_embeddings)
        text_embeddings = transform(text_embeddings)

    mapping = params.mapping
    print(f"Mapping text to sound with method {mapping}...")
    # mapping options: identity, cluster_map
    if mapping == "identity":
        mapping_fn = identity
    elif mapping == "cluster":
        mapping_fn = cluster_map
    else:
        raise Exception("Invalid mapping provided")

    neighbor_indices = mapping_fn(sound_embeddings, text_embeddings, params.distance_metric)

    print("Evaluating mapping...")
    if params.distance_metric == "euclidean":
        distance_fn = lambda x, y: np.linalg.norm(x - y)
    elif params.distance_metric == "cosine":
        distance_fn = lambda x, y: distance.cosine(x, y)

    score = evaluate_mapping(text_embeddings, sound_embeddings[neighbor_indices], distance_fn)
    print(f"Score: {score}")

    print("Fetching sounds...")
    output_sounds = [sound_corpus[i] for i in neighbor_indices]

    print("Saving output...")
    save_output(output_sounds, OUTPUT_PATH)

    print("Done.")

if __name__ == "__main__":

    e = Evaluator(
        sound_path="./corpora/sound/toy",
        text_path="./corpora/text/test.txt",
        sound_encoders=["MuQ"],
        text_encoders=["fastText"],
        mappings=["identity"],
        sound_preprocessings=[1000],
        normalizations=["standard"],
        dims=[2, 10, 30],
        distance_metrics=["euclidean", "cosine"],
        mapping_evaluations=["pairwise"]
    )

    parameter_list = e.create_params()
    for parameters in parameter_list:
        run(parameters)
