import torch
import numpy as np
from scipy.signal import resample_poly
from muq import MuQ
from transformers import AutoTokenizer, RobertaModel
import gensim.downloader as api
import fasttext
from librosa import resample


def muq(wav, sr):
    device = 'cuda'

    # Convert stereo to mono by averaging channels
    if len(wav.shape) == 2:  # Check if stereo (2D array)
        wav = wav.mean(axis=1)  # Average the two channels

    # Convert to float32 if needed and normalize to [-1, 1] range
    # if wav.dtype != np.float32:
    #     wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max

    # Resample if the sample rate is not 24000 Hz
    target_sr = 24000
    if sr != target_sr:
        wav = resample(wav, orig_sr=sr, target_sr=target_sr)  # Resample to 24000 Hz
        sr = target_sr  # Update sample rate variable

    if wav.size < 1024:  # Muq doesn't work with shorter signals
        print("Signal too short for MuQ, skipping...")
        return None

    # Convert to torch tensor and move to device
    wav = torch.tensor(wav).unsqueeze(0).to(device)

    # Load MuQ model
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq = muq.to(device).eval()

    # Run model inference
    with torch.no_grad():
        output = muq(wav, output_hidden_states=True)

    print('MuQ: feature shape: ', output.last_hidden_state.shape)

    embedding = output.last_hidden_state.squeeze() # Remove batch dimension
    # For MuQ: average across time
    embedding = torch.mean(embedding, dim=0).flatten()

    return embedding.cpu()


def RoBERTa(text):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model = RobertaModel.from_pretrained("FacebookAI/roberta-base")

    inputs = tokenizer(text, return_tensors="pt", max_length=512, padding=True, truncation=True, stride=50)
    tokens = inputs.encodings[0].tokens
    outputs = model(**inputs)
    print("RoBERTa: number of tokens: ", len(tokens))

    last_hidden_states = outputs.last_hidden_state
    # remove outputs from the start of sequence and end of sequence tokens
    last_hidden_states = last_hidden_states.squeeze()
    last_hidden_states = last_hidden_states[1:last_hidden_states.shape[0]-1]
    return last_hidden_states.detach()


def fastText(text):
    model = fasttext.load_model('fastText/cc.en.300.bin')
    embeddings = []
    for word in text.split():
        emb = model.get_word_vector(word)
        embeddings.append(emb)
    embeddings = np.stack(embeddings)
    embeddings = torch.from_numpy(embeddings)
    return embeddings


def word2vec(text):
    # Download and load the pre-trained Word2Vec model
    w2v_model = api.load('word2vec-google-news-300')
    vecs = []
    for word in text.split():
        # Retrieve the word vector
        try:
            word_vector = w2v_model[word]
            vecs.append(word_vector)
        except KeyError:
            print("Word '{}' not found in vocabulary".format(word))
            continue
    vecs = np.stack(vecs)
    print("word2vec shape: ", vecs.shape)
    return torch.from_numpy(vecs)
