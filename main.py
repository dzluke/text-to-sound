import numpy as np
from pathlib import Path
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from models import muq, RoBERTa
import librosa
import soundfile as sf
import torch

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
                # set to 10 seconds
                data = librosa.util.fix_length(data, size=10 * SAMPLING_RATE)
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


def find_nearest_neighbors(sound_embeddings, points, distance_metric):
    """

    :param sound_embeddings: The space S of all sound embeddings
    :param points: The text embeddings that have been mapped to S
    :param distance_metric: 'euclidean' or 'cosine'
    :return: neighbor_indices: list of indices that correspond to points in S
    """
    neigh = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=distance_metric).fit(sound_embeddings)
    neighbor_indices = neigh.kneighbors(points, return_distance=False).flatten()
    return neighbor_indices


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


def main():
    parameters = {
        "sound_corpus_path": "./corpora/sound/toy",
        "text_corpus_path": "./corpora/text/repeat.txt",
        "sound_encoder": "MuQ",
        "text_encoder": "RoBERTa",
        "output_path": "./output",
        "grain_size": 1000,  # in ms
        "distance": "euclidean",
        "trim_silence": True,
    }

    normalization = StandardScaler()
    dim = 10  # the number of dimensions to reduce to

    # a function that determines how to separate sounds
    grain_size = int(parameters['grain_size'] / 1000.0 * SAMPLING_RATE)  # convert to samples
    slice_fn = lambda y: equal_slices(y, grain_size)

    sound_corpus_path = Path(parameters["sound_corpus_path"])
    text_corpus_path = Path(parameters["text_corpus_path"])

    if parameters["sound_encoder"] == "MuQ":
        sound_encoder = muq
    if parameters["text_encoder"] == "RoBERTa":
        text_encoder = RoBERTa

    print("Loading sound and text data...")
    sound_corpus = load_soundfiles(sound_corpus_path)
    sound_corpus = preprocess_sounds(sound_corpus, slice_fn, parameters["trim_silence"])

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

    print("Mapping text to sound...")
    W = np.eye(dim)
    mapped_text_embeddings = [W @ emb for emb in text_embeddings]

    print("Finding nearest neighbors...")
    neighbor_indices = find_nearest_neighbors(sound_embeddings, mapped_text_embeddings, parameters["distance"])

    print("Fetching sounds...")
    output_sounds = [sound_corpus[i] for i in neighbor_indices]

    print("Saving output...")
    save_output(output_sounds, Path(parameters["output_path"]))

    print("Done.")

if __name__ == "__main__":
    main()

