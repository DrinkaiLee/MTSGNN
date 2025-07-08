import numpy as np


def change_dataset_format(data, labels=None, include_subjects: bool = False):
    """
    Change dataset format and provide corresponding labels.

    :param data: array, data to be reformatted. shape=([subjects], trials, channels, samples) or (channels, samples, targets, trials_per_target, [subjects]).
    :param labels: array(1-d), data labels, required only when data shape=([subjects], trials, channels, samples).
    :param include_subjects: bool, whether data contains subjects dimension.
    :return: list, [reformatted data, corresponding labels]
    """
    if include_subjects:
        dim_data = len(np.shape(data))
        assert dim_data in [4, 5], f"The shape of the dataset does not match, it must be 4-d or 5-d."

        if dim_data == 4:
            dim_labels = len(np.shape(labels))

            num_subjects, num_trials, num_channels, num_samples = np.shape(data)

            formatted_data = []
            formatted_labels = []
            for subject_i in range(num_subjects):
                data_i = data[subject_i, :, :, :]
                labels_i = labels if dim_labels == 1 else labels[subject_i, :]
                formatted_results = _change_dataset_format(data=data_i, labels=labels_i)
                formatted_data.append(formatted_results[0])
                formatted_labels.append(formatted_results[1])

            formatted_data = np.array(formatted_data).astype(formatted_data[0].dtype)
            formatted_data = np.transpose(formatted_data, (1, 2, 3, 4, 0))
            formatted_labels = np.array(formatted_labels).astype(formatted_labels[0].dtype)
            formatted_labels = np.transpose(formatted_labels, (1, 2, 0))
            return formatted_data, formatted_labels

        elif dim_data == 5:
            num_channels, num_samples, num_targets, num_trials_per_target, num_subjects = np.shape(data)

            formatted_data = []
            formatted_labels = []
            for subject_i in range(num_subjects):
                data_i = data[:, :, :, :, subject_i]
                formatted_results = _change_dataset_format(data=data_i)
                formatted_data.append(formatted_results[0])
                formatted_labels.append(formatted_results[1])
            formatted_data = np.array(formatted_data).astype(formatted_data[0].dtype)
            formatted_labels = np.array(formatted_labels).astype(formatted_labels[0].dtype)
            return formatted_data, formatted_labels

    else:
        return _change_dataset_format(data=data, labels=labels)


def _change_dataset_format(data, labels=None):
    """
    Change dataset format and provide corresponding labels.
    3-d data: (trials, channels, samples) -> (channels, samples, targets, trials_per_target)
    4-d data: (channels, samples, targets, trials_per_target) -> (trials, channels, samples)

    :param data: array(3-d or 4-d), data to be reformatted.
    :param labels: array(1-d), data labels, required only when data is 3-d.
    :return: list, [reformatted data, corresponding labels]
    """
    dim_data = len(np.shape(data))
    assert dim_data in [3, 4], f"The shape of the dataset does not match, it must be 3-d or 4-d."

    if dim_data == 3:
        num_trials, num_channels, num_samples = np.shape(data)
        assert len(np.shape(labels)) == 1, f"The label shape does not match, it must be 1-d."
        assert len(labels) == num_trials, f"The length of label is not equal to data.shape[0]."

        unique_label, counts_label = np.unique(labels, return_counts=True)
        assert np.all(np.equal(counts_label, counts_label[0])), \
            f"The number of labels needs to be the same for different categories."
        num_targets = len(np.unique(labels))
        num_trials_per_target = counts_label[0]

        # Convert 3-d data to 4-d format
        formatted_data = np.zeros((num_channels, num_samples, num_targets, num_trials_per_target))
        for target_i in range(num_targets):
            target_i_index = np.where(labels == unique_label[target_i])[0]
            target_i_data = data[target_i_index, :, :]
            for trial_i in range(num_trials_per_target):
                formatted_data[:, :, target_i, trial_i] = target_i_data[trial_i, :, :]
        formatted_labels = get_labels(formatted_data)

    else:
        num_channels, num_samples, num_targets, num_trials_per_target = np.shape(data)

        formatted_labels = get_labels(data)
        formatted_labels = formatted_labels.reshape(-1)
        formatted_data = np.reshape(data, (num_channels, num_samples, -1))
        formatted_data = np.transpose(formatted_data, axes=(2, 0, 1))
    return [formatted_data, formatted_labels]


def get_labels(data):
    """
    Generate labels for data with shape=(channels, samples, targets, trials, [subjects]).

    :param data: array(4-d or 5-d), shape=(channels, samples, targets, trials, [subjects])
    :return: array(2-d or 3-d), corresponding labels, shape=(targets, trials, [subjects])
    """
    dim_data = len(np.shape(data))
    assert dim_data in [4, 5], f"The shape of the dataset does not match, it must be 4-d or 5-d."
    if dim_data == 4:
        _, _, num_targets, num_trials = np.shape(data)
        labels = np.zeros(shape=(num_targets, num_trials), dtype=int)
        for target_i in range(num_targets):
            labels[target_i, :] = target_i

    else:
        _, _, num_targets, num_trials, num_subjects = np.shape(data)
        labels = np.zeros(shape=(num_targets, num_trials, num_subjects), dtype=int)
        for target_i in range(num_targets):
            labels[target_i, :, :] = target_i

    return labels
