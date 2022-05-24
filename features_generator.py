"""
    Processing & feature extraction
    --please specify the raw data directory (raw_dir) and the output directory (tar_dir)
"""

import numpy as np
import os
import pickle
import functions as func


# -----------------------params-----------------------
raw_dir = ''
tar_dir = 'Processed'
print(f'read data from {raw_dir}')
print(f'save extracted features to {tar_dir}')

window_width = 96
stride_len = window_width // 2
cut_off_freq = 11.0
sample_freq = 100.0
subjects = 9
target_filename = os.path.join(tar_dir, f'features_w{window_width}.pkl')

# -----------preprocess & feature extract--------------
for subject in range(1, subjects+1):
    first_time = True

    # get rid of subject 9
    if subject == 9:
        continue

    file = f'subject10{subject}.dat'
    print(f'{file} is reading & preprocessing...')
    x, y = func.data_preprocess(os.path.join(raw_dir, file), mag_used=True)

    # median filter
    data_raw = func.median(x, kernel_size=5)
    # lowpass filter
    data_raw = func.low_pass(data_raw, fn=cut_off_freq, fs=sample_freq)

    # sliding window
    temp_y = []
    number = (len(data_raw) - window_width) // stride_len + 1
    for w in range(number):
        w_start = w * stride_len
        w_end = w_start + window_width
        temp_raw = data_raw[w_start:w_end]
        temp_y.append(y[w_start + window_width // 2])

        # domain extension
        t_data = temp_raw
        f_data = func.fast_fourier_transform(t_data)
        mag_t_data = func.magnitude_of_triaxial(t_data)
        f_mag_t_data = func.fast_fourier_transform(mag_t_data)

        # features extract
        t_features = func.t_features_generate(t_data)
        mag_t_features = func.mag_t_features_generate(mag_t_data)
        f_features = func.f_features_generate(f_data, sample_freq)
        f_mag_t_features = func.f_mag_t_features_generate(f_mag_t_data, sample_freq)
        feature_vector = t_features + mag_t_features + f_features + f_mag_t_features

        if first_time:
            first_time = False
            features = np.array(feature_vector).reshape((1, -1))
        else:
            features = np.vstack((features, np.array(feature_vector).reshape((1, -1))))

    features_y = np.hstack((features, np.array(temp_y).reshape(-1, 1)))
    np.savetxt(os.path.join(tar_dir, f'subject10{subject}.txt'), features_y, fmt='%.6f', delimiter=', ')

# -----------------------feature names-----------------------
feature_names = func.t_features_names() + func.mag_t_features_names() + \
                func.f_features_names() + func.f_mag_features_names()
print(f'total features number: {len(feature_names)}')
with open(os.path.join(tar_dir, 'feature_names.txt'), 'w') as f:
    for name in feature_names:
        f.write(name + '\n')

# ------------collect and save features by pickle---------------
data_x = np.empty((0, len(feature_names)))
data_y = np.empty(0)
data_group = np.empty(0)

for subject in range(1, subjects+1):

    if subject == 9:
        continue

    data = np.loadtxt(os.path.join(tar_dir, f'subject10{subject}.txt'), delimiter=',')
    features = data[:, :-1]
    labels = data[:, -1]
    data_x = np.vstack((data_x, features))
    data_y = np.concatenate((data_y, labels))
    data_group = np.concatenate((data_group, np.ones(len(data)) * subject))

print(data_x.shape, data_y.shape, data_group.shape)

obj = (data_x, data_y, data_group)
with open(target_filename, 'wb') as f:
    pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)
