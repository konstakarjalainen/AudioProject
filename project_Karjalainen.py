import numpy as np

import librosa as lb
import librosa.display

from numpy.fft import fft, ifft

import sounddevice as sd

from scipy.signal import hann

from matplotlib import pyplot as plt


def calc_delta(a, h, p, d):
    rows = h.shape[0]-1
    cols = h.shape[1]-1

    for i in np.arange(0, cols):
        for j in np.arange(0, rows):
            d[j, i] = a * (h[j][i-1] - 2*(h[j][i]) + h[j][i+1])/4 \
                        - (1 - a)*(p[j-1][i] - 2*(p[j][i]) + p[j+1][i])/4

    return d


def idft(ffts, A, win_size, overlap):
    a = np.zeros((A.shape[0] * A.shape[1])//2)  # Placeholder
    num_frames = A.shape[1]
    first_frame = A[:, 0]
    first_frame = np.power(first_frame, (1/(2*gamma)))*np.exp(1j*np.angle(ffts[:, 0]))
    first_recon = ifft(first_frame).real
    first_win = first_recon * window
    a[0:win_size] = first_win
    for i in np.arange(1, num_frames-1):
        frame = A[:, i]
        frame = np.power(frame, (1/(2*gamma)))*np.exp(1j*np.angle(ffts[:, i]))
        frame_recon = ifft(frame).real
        frame_win = frame_recon * window
        a[i*overlap:i*overlap + win_size] = a[i*overlap:i*overlap + win_size] + frame_win
    return a


audio, sr = lb.load('project_test1.wav', sr=16000)


win_size = 1024
window = hann(win_size, sym=False)
hop_size = win_size//2
n_fft = win_size
n_frames = int((len(audio)-win_size)/hop_size)+1

gamma = 0.3  # From supporting material
ffts = np.zeros((n_fft, n_frames), dtype=np.complex_)
spectrogram = np.zeros((n_fft//2, n_frames), dtype=np.complex_)

# STFT framewise
for i in np.arange(0, n_frames):
    s = audio[i * hop_size:i * hop_size + win_size] * window
    spectrum = fft(s, n_fft)
    ffts[:, i] = spectrum
    spectrum = spectrum[:n_fft//2]
    spectrogram[:, i] = spectrum

power_spectrogram = np.power(np.abs(ffts), 2*gamma)
k_max = 10  # Num of iterations
k = 0
W = power_spectrogram
H = power_spectrogram/2
P = power_spectrogram/2
alpha = np.var(P)/(np.var(H)+np.var(P))

# Iteration
while k < k_max-1:
    delta = np.zeros((n_fft, n_frames), dtype=np.float32)
    delta = calc_delta(alpha, H, P, delta)
    H = H + delta
    H[H < 0] = 0
    np.copyto(H, W, where=W < H)
    P = W - H
    k += 1

H_binarized = np.multiply(H >= P, W)
P_binarized = np.multiply(H < P, W)

h = idft(ffts, H_binarized, win_size, hop_size)
p = idft(ffts, P_binarized, win_size, hop_size)

spec_h = lb.amplitude_to_db(np.abs(lb.stft(h)), ref=np.max)
spec_p = lb.amplitude_to_db(np.abs(lb.stft(p)), ref=np.max)

fig, ax = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(10)
img = lb.display.specshow(spec_h, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='harmonic')
fig.colorbar(img, ax=ax)

fig, ax = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(10)
img = lb.display.specshow(spec_p, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='percussive')
fig.colorbar(img, ax=ax)

fig, ax = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(10)
img = lb.display.specshow(lb.amplitude_to_db(np.abs(spectrogram)), x_axis='time', y_axis='linear', ax=ax)
ax.set(title='original signal')
fig.colorbar(img, ax=ax)
plt.show()

separated = h+p
# For taking stft in loop, separated audio got shorter
audio_resized = audio[:separated.size]
noise = audio_resized-separated
SNR = 10*np.log10(sum(audio_resized**2)/sum(noise**2))
print("Signal-to-noise ratio is {:2.2f} dB".format(SNR))

sd.play(h, sr)
sd.wait()

sd.play(p, sr)
sd.wait()

sd.play(separated, sr)
sd.wait()

