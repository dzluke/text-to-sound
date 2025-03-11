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
    output = np.concatenate(np.array(sound_list))

    output_path.mkdir(parents=True, exist_ok=True)
    counter = 0
    while (output_path / f"output{counter}.wav").exists():
        counter += 1

    filename = output_path / f"output{counter}.wav"
    sf.write(filename, output, SAMPLING_RATE)


def main():
    parameters = {
        "sound_corpus_path": "./corpora/sound/anonymous_corpus",
        "text_corpus_path": "./corpora/text/test.txt",
        "sound_encoder": "MuQ",
        "text_encoder": "RoBERTa",
        "output_path": "./output",
        "sound_length": 10,
        "distance": "euclidean",
    }

    normalization = StandardScaler()
    dim = 2  # the number of dimensions to reduce to

    sound_corpus_path = Path(parameters["sound_corpus_path"])
    text_corpus_path = Path(parameters["text_corpus_path"])

    if parameters["sound_encoder"] == "MuQ":
        sound_encoder = muq
    if parameters["text_encoder"] == "RoBERTa":
        text_encoder = RoBERTa

    print("Loading sound and text data...")
    sound_corpus = load_soundfiles(sound_corpus_path)
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

