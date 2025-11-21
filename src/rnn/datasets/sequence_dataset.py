import numpy as np
from src.rnn.features.seq_features import compute_seq_features
from src.common.io.dummy_data import generate_dummy_emg
from src.common.io.schemas import load_cfg
from src.common.preprocessing.windowing import window_signal

