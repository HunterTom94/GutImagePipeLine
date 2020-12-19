from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

max_ls = []
slope_ls = []
for i in range(100000):
    t = np.linspace(1, 23, 23)
    # x_volts = 1*np.sin(t/(2*np.pi))
    x_volts = np.ones_like(t)

    x_watts = x_volts ** 2
    # Set a target SNR
    target_snr_db = 20
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    # Noise up the original signal
    y_volts = x_volts + noise_volts

    # max_ls.append(np.max(y_volts))
    slope_ls.append(np.max(np.gradient(y_volts)))

sns.distplot(slope_ls)
#
# plt.plot(t, y_volts)
plt.show()