import torch
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from muq import MuQ
from transformers import AutoTokenizer, RobertaModel


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
        wav = resample_poly(wav, up=target_sr, down=sr)  # Resample to 24000 Hz
        sr = target_sr  # Update sample rate variable

    # Convert to torch tensor and move to device
    wavs = torch.tensor(wav).unsqueeze(0).to(device)

    # Load MuQ model
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq = muq.to(device).eval()

    # Run model inference
    with torch.no_grad():
        output = muq(wavs, output_hidden_states=True)

    print('MuQ: feature shape: ', output.last_hidden_state.shape)

    return output.last_hidden_state.cpu()


def RoBERTa(text):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model = RobertaModel.from_pretrained("FacebookAI/roberta-base")

    inputs = tokenizer(text, return_tensors="pt", max_length=512, padding=True, truncation=True, stride=50)
    tokens = inputs.encodings[0].tokens
    outputs = model(**inputs)
    print("RoBERTa: number of tokens: ", len(tokens))

    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.detach().squeeze()
