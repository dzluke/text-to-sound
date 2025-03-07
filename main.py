import numpy as np
from scipy.io import wavfile
from pathlib import Path
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA

SAMPLING_RATE = 44100

parameters = {
    "sound_corpus_path": "./sound_corpus",
    "text_corpus_path" : "./text_corpus",
    "sound_encoder": "",
    "text_encoder": "",
    "output_path": "./output.wav",
}


sound_corpus_path = Path(parameters["sound_corpus_path"])
text_corpus_path = Path(parameters["text_corpus_path"])

# a function that takes in a sound or word and returns an embedding
sound_encoder = lambda x: np.random.rand(128)
sound_decoder = lambda x: np.random.rand(44100 * 3)
text_encoder = lambda x: np.random.rand(500)

# the function that will apply dimensionality reduction
dim = 30
pca = PCA(n_components=dim)
dim_reduction = pca.fit_transform

# what transformations will we apply to the feature spaces?
transform_pipeline = [Normalizer, dim_reduction]

# how do we find the nearest neighbor in the sound space?
nearest_neighbor = lambda x: x

sound_corpus = []
for file in sound_corpus_path.iterdir():
    if file.suffix in [".wav", ".aif", ".mp3"]:
        sr, data = wavfile.read(file)
        assert sr == SAMPLING_RATE
        sound_corpus.append(data)

text_corpus = []
for file in text_corpus_path.iterdir():
    with open(file, "r") as f:
        for line in f.readlines():
            for word in line.split():
                text_corpus.append(word)

sound_embeddings = []
for sound in sound_corpus:
    embedding = sound_encoder(sound)
    sound_embeddings.append(embedding)

text_embeddings = []
for text in text_corpus:
    embedding = text_encoder(text)
    text_embeddings.append(embedding)

# transform the embedding spaces
for transform in transform_pipeline:
    sound_embeddings = transform(sound_embeddings)
    text_embeddings = transform(text_embeddings)

# the mapping between spaces is defined by the matrix W
W = np.eye(dim)

# for each word input, find the resulting sound
mapped_text_embeddings = []
for word_embedding in text_embeddings:
    sound_embedding = W @ word_embedding
    mapped_text_embeddings.append(sound_embedding)

# for each mapped text embedding, find the nearest sound
output_sound_embeddings = []
for text_embedding in mapped_text_embeddings:
    s = nearest_neighbor(text_embedding)
    output_sound_embeddings.append(s)

# convert sound embeddings to sounds
output_sounds = []
output_sound_length = 0
for sound_embedding in output_sound_embeddings:
    sound = sound_decoder(sound_embedding)
    output_sounds.append(sound)
    output_sound_length += sound.shape[0] # assuming these are mono sounds

# concatenate sounds
output = np.concatenate(output_sounds)

# write to file
wavfile.write(parameters["output_path"], SAMPLING_RATE, output)
