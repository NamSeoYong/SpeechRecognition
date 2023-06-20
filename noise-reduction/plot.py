import librosa
import matplotlib.pyplot as plt

sample_rate = 16000
denoise, _ = librosa.load(data/handae.wav, sr=sample_rate)
noisy, _ = librosa.load(denoise_data/de_handae.wav, sr=sample_rate)
plt.figure()
plt.plot(denoise, label='denoise')
plt.plot(noisy, label='noisy', alpha=0.5)
plt.legend()
plt.show()