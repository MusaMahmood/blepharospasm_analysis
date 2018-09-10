import numpy as np
from scipy.io import savemat

from tf_shared_k import prep_dir, load_data_v2


def ind2vec(x, dimensions=3, number_classes=None):
    if dimensions == 2:
        print('Not yet supported')
    elif dimensions == 3:
        samples = x.shape[0]
        seq_len = x.shape[1]
        if np.min(x) == 1:
            x = np.subtract(x, 1)
        if number_classes is None:
            number_classes = int(np.max(x) + 1)
        new_vec_array = np.zeros([samples, seq_len, number_classes], dtype=np.int32)
        for i in range(0, samples):
            for s in range(0, seq_len):
                new_vec_array[i, s, int(x[i, s, 0])] = 1
        return new_vec_array
    else:
        print('Not yet supported')
    return 0


num_classes = 1
seq_length = 2000
input_length = seq_length
dir_x = 'data_labeled_3c/'
file_location = prep_dir(dir_x + '_all/') + 'all_data.mat'  # Output Directory
key_x = 'X'
key_y = 'Y'

# Load Data From Folder dir_x
x_data, y_data = load_data_v2(dir_x, [seq_length, 1], [seq_length, 1], 'relevant_data', 'Y')

# Change labels to vectors: (n, 1) → (n, num_classes)
y_data = ind2vec(y_data, dimensions=3)
print("Loaded Data Shape: X:", x_data.shape, " Y: ", y_data.shape)

# Save Matlab Output File:
savemat(file_location, mdict={key_x: x_data, key_y: y_data})
print("Saved Data: KeyX:", key_x, " KeyY: ", key_y, ' at: ', file_location)
