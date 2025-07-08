import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import zscore
from sklearn.model_selection import KFold
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint

from preprocess import change_dataset_format
from Models import MTSGNN, EEGNet_v1, EEGNet_v3


# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

# Select model
model_name = sys.argv[2]
assert model_name in ['mtsgnn', 'eegnetv1', 'eegnetv3'], \
    "Model name must be either 'mtsgnn', 'eegnetv1' or 'eegnetv3'."
if model_name == 'mtsgnn':
    network = MTSGNN
elif model_name == 'eegnetv1':
    network = EEGNet_v1
elif model_name == 'eegnetv3':
    network = EEGNet_v3
print(f"Using model: {model_name}")

# Dataset and results path
current_path = Path(os.path.dirname(os.path.abspath(__file__)))
dataset_path = current_path / 'example_dataset.mat'
checkpoint_folder_path = current_path / 'checkpoint'
results_folder_path = current_path / 'results'
for folder in [checkpoint_folder_path, results_folder_path]:
    if not os.path.exists(folder):
        os.makedirs(folder)
results_path = results_folder_path / f'results_{model_name}.txt'

# Dataset parameters
fs = 256                # sample rate
num_classes = 4         # number of categories
num_block = 64          # number of block
cue_time = 1            # cue time in seconds
offset_time = 0.14      # visual offset time in seconds
duration_time = 2       # duration of each trial in seconds
start_index = round((cue_time + offset_time) * fs)
end_index = start_index + round(duration_time * fs)

# Training parameters
k_split = 10            # K-Fold
epochs = 500            # epoch
batch_size = 32         # batch size
val_split = 0.2         # validation set ratio

# Split the training and test blocks
train_block_list = []
test_block_list = []
kf = KFold(n_splits=k_split, shuffle=True, random_state=42)
for fold_i, (train_index, test_index) in enumerate(kf.split([i for i in range(num_block)])):
    train_block_list.append(train_index)
    test_block_list.append(test_index)
    
# Load dataset
data = sio.loadmat(dataset_path)['data'][:, :, start_index:end_index]
label = sio.loadmat(dataset_path)['label'].reshape(-1)
data = zscore(data, axis=-1)
data, label = change_dataset_format(data, label)

# ---------------------------- Training Stage ----------------------------
f = open(results_path, 'w')
av_acc = 0
for fold_i in range(k_split):
    print(f'-------------- Fold {fold_i + 1}/{k_split} --------------')
    print('Train blocks:', train_block_list[fold_i])
    print('Test blocks:', test_block_list[fold_i])
    
    # Split dataset
    x_train, x_test = data[:, :, :, train_block_list[fold_i]], data[:, :, :, test_block_list[fold_i]]
    y_train, y_test = label[:, train_block_list[fold_i]], label[:, test_block_list[fold_i]]

    x_train, y_train = change_dataset_format(x_train, y_train)
    x_test, y_test = change_dataset_format(x_test, y_test)
    
    x_train = x_train.reshape(*x_train.shape, 1)
    x_test = x_test.reshape(*x_test.shape, 1)
            
    [_, chans, samples, _] = x_train.shape
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    # Shuffle
    shuffle_index = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_index)
    x_train = x_train[shuffle_index, :, :]
    y_train = y_train[shuffle_index, :]

    # Initialize
    model = network(nb_classes=num_classes, Chans=chans, Samples=samples)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpoint_path = checkpoint_folder_path / f'{model_name}_fold_{fold_i + 1}.h5'
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=0,
                                   save_best_only=True, monitor='val_accuracy', mode='max')
    
    # Train
    fittedModel = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                            verbose=0, validation_split=val_split, callbacks=[checkpointer], shuffle=True)
    
    # Evaluate
    acc = model.evaluate(x_test, y_test)[-1]
    av_acc += acc
    
    # Log
    print(f'Fold {fold_i + 1} accuracy: {acc:.4f}')
    f.write(f'Fold {fold_i + 1} accuracy: {acc:.4f}\n')

print(f'Average accuracy: {av_acc / k_split:.4f}')
f.write(f'Average accuracy: {av_acc / k_split:.4f}\n')
f.close()
