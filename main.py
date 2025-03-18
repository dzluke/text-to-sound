import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
import torch
from itertools import combinations
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from scipy.spatial import distance

from models import muq, RoBERTa, word2vec, fastText
from util import Parameter, ParameterGenerator, load_soundfiles, remove_silence, set_sampling_rate


SAMPLING_RATE = 44100
OUTPUT_PATH = Path("./output")
CACHE_PATH = Path("./cache")

set_sampling_rate(SAMPLING_RATE)


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
    

def generate_sound_cache_filename(corpus_name, slice_fn_name, encoder_name):
    """
    Generate a unique filename for sound embeddings based on corpus name, slice function, and encoder name.
    """
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return CACHE_PATH / f"sound_{corpus_name}_{slice_fn_name}_{encoder_name}.pkl"

def generate_text_cache_filename(text_file_name, encoder_name):
    """
    Generate a unique filename for text embeddings based on text file name and encoder name.
    """
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return CACHE_PATH / f"text_{text_file_name}_{encoder_name}.pkl"


def embed_sounds(sounds, encoder, cache_file=None):
    # Check if cache_file is provided and exists
    if cache_file is not None and cache_file.exists():
        print(f"Loading sound embeddings from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            embeddings = pickle.load(f)
            return embeddings
    
    # Batch process sounds
    embeddings = encoder(sounds, SAMPLING_RATE)
    
    if cache_file is not None:
        # Save embeddings to cache
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
    
    return embeddings


def embed_text(text, encoder, cache_file=None):
    # if cache_file is provided, load the embeddings from there
    if cache_file is not None and cache_file.exists():
        print(f"Loading text embeddings from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    embeddings = encoder(text)

    if cache_file is not None:
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)

    return embeddings


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


def save_output(sound_list, filename):
    # concatenate audio files
    output = np.concatenate(sound_list)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    counter = 1
    while (filename).exists():
        filename = filename.parent / f"{filename.stem}_{counter}{filename.suffix}"
        counter += 1

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


def pairwise_score(t_embs, s_embs, distance_metric):
    """
    s_embs[i] is the sound mapped from t_embs[i]
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
    return nearest_sound_idx, (sound_cluster_labels, text_cluster_labels)  # return the cluster labels for evaluation purposes


def run(params, evaluator=None, cache=True):
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
        slice_fn_name = f"grain{params.sound_preprocessing}"
    except TypeError:
        if params.sound_preprocessing == "onsets":
            slice_fn = lambda y: get_onsets(y)
            slice_fn_name = "onsets"
        elif params.sound_preprocessing == "full":
            # slice_fn = lambda y: y
            slice_fn_name = "full"
        else:
            raise ValueError(f"Unknown preprocessing {params.sound_preprocessing}")
    if params.sound_preprocessing != 'full':
        sound_corpus = preprocess_sounds(sound_corpus, slice_fn, params.trim_silence)

    if params.sound_encoder == "MuQ":
        sound_corpus = [s for s in sound_corpus if s.size > 1024]  # MuQ requires sounds longer than 1024 samples

    sound_cache_file = None
    text_cache_file = None
    if cache:
        # Generate cache file paths
        sound_cache_file = generate_sound_cache_filename(
            corpus_name=sound_corpus_path.stem,
            slice_fn_name=slice_fn_name,
            encoder_name=params.sound_encoder,
        )
        text_cache_file = generate_text_cache_filename(
            text_file_name=text_corpus_path.name,
            encoder_name=params.text_encoder,
        )

    print("Embedding sounds...")
    sound_embeddings = embed_sounds(sound_corpus, sound_encoder, sound_cache_file)

    print("Embedding text...")
    text_embeddings = embed_text(" ".join(text_corpus), text_encoder, text_cache_file)

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
        neighbor_indices = identity(sound_embeddings, text_embeddings, params.distance_metric)
    elif mapping == "cluster":
        neighbor_indices, (sound_cluster_labels, text_cluster_labels) = cluster_map(sound_embeddings, text_embeddings, params.distance_metric)
    else:
        raise Exception("Invalid mapping provided")

    print("Evaluating mapping...")
    if params.distance_metric == "euclidean":
        distance_fn = lambda x, y: np.linalg.norm(x - y)
    elif params.distance_metric == "cosine":
        distance_fn = lambda x, y: distance.cosine(x, y)

    score = pairwise_score(text_embeddings, sound_embeddings[neighbor_indices], distance_fn)
    print(f"Score: {score}")

    print("Fetching sounds...")
    output_sounds = [sound_corpus[i] for i in neighbor_indices]

    print("Saving output...")
    save_path = OUTPUT_PATH / f"{params.filename()}.wav"
    save_output(output_sounds, save_path)

    print("Done.")

    # Save scores if evaluator is provided
    if evaluator:
        scores = {"pairwise_score": score}

        if params.mapping == "cluster":
            # First, evaluate individual spaces
            sound_eval = evaluate_clustering(sound_embeddings, sound_cluster_labels)
            text_eval = evaluate_clustering(text_embeddings, text_cluster_labels)
            
            # Add prefixes to distinguish between sound and text metrics
            sound_scores = {f"sound_{key}": value for key, value in sound_eval.items()}
            text_scores = {f"text_{key}": value for key, value in text_eval.items()}
            scores.update(sound_scores)
            scores.update(text_scores)
            
            # Now evaluate the combined space
            # 1. Create combined embeddings
            combined_embeddings = np.vstack((sound_embeddings, text_embeddings))
            
            # 2. Create combined labels 
            # - Keep original cluster assignments
            # - Offset text clusters to avoid overlap (e.g., if sound has clusters 0,1,2, text would be 3,4,5)
            num_sound_clusters = np.max(sound_cluster_labels)
            combined_labels = np.concatenate([
                sound_cluster_labels,
                text_cluster_labels + num_sound_clusters + 1
            ])
            
            # 3. Also create domain labels (0 for sound, 1 for text)
            domain_labels = np.concatenate([
                np.zeros(len(sound_embeddings)),
                np.ones(len(text_embeddings))
            ])
            
            # 4. Evaluate combined space with cluster labels
            combined_cluster_eval = evaluate_clustering(combined_embeddings, combined_labels)
            combined_scores = {f"combined_{key}": value for key, value in combined_cluster_eval.items()}
            scores.update(combined_scores)
            
            # 5. Evaluate separation between domains
            domain_eval = evaluate_clustering(combined_embeddings, domain_labels)
            domain_scores = {f"domain_{key}": value for key, value in domain_eval.items()}
            scores.update(domain_scores)
        
        evaluator.save_result(params, scores)

        return scores

if __name__ == "__main__":

    e = ParameterGenerator(
        sound_path="./corpora/sound/toy",
        text_path="./corpora/text/test.txt",
        sound_encoders=["MuQ"],
        text_encoders=["fastText"],
        mappings=["identity"],
        sound_preprocessings=['full'],
        normalizations=["standard"],
        dims=[2, 10, 30],
        distance_metrics=["euclidean", "cosine"],
        mapping_evaluations=["pairwise"]
    )

    parameter_list = e.create_params()
    for parameters in parameter_list:
        scores = run(parameters, cache=False)
