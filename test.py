import numpy as np
import scipy.io
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from scipy.signal import welch


exp_type = ['Arithmetic_', 'Mirror_image_', 'Relax_', 'Stroop_']
eeg_data = []
alpha_band = (8, 12)
beta_band = (13, 30)
alpha_features = []
beta_features = []
power_ratio = []
features = np.zeros((480, 32))
count = 0

for ii in range(4):
    for jj in range(1, 41):
        for kk in range(1, 4):
            file_name = f"{exp_type[ii]}sub_{jj}_trial{kk}.mat"
            print(file_name)
            mat = scipy.io.loadmat(file_name)
            print('--------------------------------------')
            eeg_data = mat["Clean_data"]
            sampling_rate = 1000
            frequencies, psd = welch(eeg_data, fs=sampling_rate, nperseg=1024)
            alpha_power = np.trapz(psd[:, (frequencies >= alpha_band[0]) & (
                frequencies <= alpha_band[1])], axis=1)
            beta_power = np.trapz(
                psd[:, (frequencies >= beta_band[0]) & (frequencies <= beta_band[1])], axis=1)
            alpha_features.append(alpha_power)
            beta_features.append(beta_power)
            alpha = np.trapz(psd[:, (frequencies >= alpha_band[0]) & (
                frequencies <= alpha_band[1])], axis=1)
            beta = np.trapz(psd[:, (frequencies >= beta_band[0]) & (
                frequencies <= beta_band[1])], axis=1)
            alpha_beta_ratio = alpha / beta
            power_ratio.extend(alpha_beta_ratio)
            features[count, :] = alpha_beta_ratio
            count = count + 1

features_df = pd.DataFrame(features)
csv_file_path = 'features.csv'
# Save the DataFrame to a CSV file
features_df.to_csv(csv_file_path, index=False)
# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)
# print(features)
