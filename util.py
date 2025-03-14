import librosa
import numpy as np
from itertools import product
from main import SAMPLING_RATE

class Parameter:
    """
    A set of parameters for one run of the system
    """
    def __init__(self, sound_path, text_path, sound_encoder, text_encoder, mapping, sound_preprocessing, normalization, dim, distance_metric, k=None, trim_silence=True):
        self.sound_path = sound_path
        self.text_path = text_path
        self.sound_encoder = sound_encoder
        self.text_encoder = text_encoder
        self.mapping = mapping
        self.sound_preprocessing = sound_preprocessing
        self.normalization = normalization
        self.dim = dim
        self.distance_metric = distance_metric
        if k is not None:
            self.k = k
        self.trim_silence = trim_silence

class Evaluator:
    """
    A set of all possible parameters to test for the system. It outputs a list of Parameter objects that exhaustively cover
    all possible input parameters.
    """

    def __init__(self, sound_path, text_path, sound_encoders, text_encoders, mappings, sound_preprocessings, normalizations, dims, distance_metrics, mapping_evaluations, ks=None, clustering_evaluations=None):
        self.sound_path = sound_path
        self.text_path = text_path
        self.sound_encoders = sound_encoders
        self.text_encoders = text_encoders
        self.mappings = mappings
        self.sound_preprocessings = sound_preprocessings
        self.normalizations = normalizations
        self.dims = dims
        self.distance_metrics = distance_metrics
        self.mapping_evaluations = mapping_evaluations
        if ks is not None:
            self.ks = ks
        if clustering_evaluations is not None:
            self.clustering_evaluations = clustering_evaluations

    # def create_params(self):
        # params = []
        # for sound_encoder in self.sound_encoders:
        #     for text_encoder in self.text_encoders:
        #         for mapping in self.mappings:
        #             for sound_preprocessing in self.sound_preprocessings:
        #                 for normalization in self.normalizations:
        #                     for dim in self.dims:
        #                         for distance_metric in self.distance_metrics:
        #                             p = Parameter(self.sound_path, self.text_path, sound_encoder, text_encoder, mapping, sound_preprocessing, normalization, dim, distance_metric)
        #                             params.append(p)

    def create_params(self):
        param_combinations = product(
            self.sound_encoders,
            self.text_encoders,
            self.mappings,
            self.sound_preprocessings,
            self.normalizations,
            self.dims,
            self.distance_metrics
        )

        params = [
            Parameter(
                self.sound_path,
                self.text_path,
                sound_encoder,
                text_encoder,
                mapping,
                sound_preprocessing,
                normalization,
                dims,
                distance_metric
            )
            for sound_encoder, text_encoder, mapping, sound_preprocessing, normalization, dims, distance_metric in param_combinations
        ]

        return params


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