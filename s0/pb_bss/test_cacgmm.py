import io
import functools

import numpy as np

import soundfile as sf
import matplotlib
import matplotlib.pylab as plt
from IPython.display import display, Audio

from einops import rearrange

from nara_wpe.utils import stft, istft

from pb_bss.distribution import CACGMMTrainer
from pb_bss.permutation_alignment import DHTVPermutationAlignment, OraclePermutationAlignment
from pb_bss.evaluation import InputMetrics, OutputMetrics

signal = []
ob = '/yrfs2/cv1/hangchen2/data/misp2021_eval/audio/eval/far/R53_S286287288289_C04_I1_Far_0.wav'
data, fs = sf.read(ob)
signal.append(data[:1000])
ob = '/yrfs2/cv1/hangchen2/data/misp2021_eval/audio/eval/far/R53_S286287288289_C04_I1_Far_1.wav'
data, fs = sf.read(ob)
signal.append(data[:1000])
data = np.stack(signal, axis=0)
obstft = stft(data, 512, 128)
# obstft = np.reshape(obstft, (1, -1, 257))
print(obstft.shape)
# plot_stft(obstft[0].T)

trainer = CACGMMTrainer()
Observation_mm = rearrange(obstft, 'd t f -> f t d')

model = trainer.fit(
    Observation_mm,
    num_classes=3,
    iterations=20,
    inline_permutation_aligner=None
)
model

affiliation = model.predict(Observation_mm)
print(affiliation)
print('*******************')
print(affiliation.shape)