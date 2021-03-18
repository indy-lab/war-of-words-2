import argparse
import os
import pickle
from collections import Counter

import numpy as np
from warofwords import Dataset


def add_text_features(features, dim):
    # Edit text features.
    for d in range(dim):
        features.add(f'edit-dim-{d}', group='edit-embedding')
    # Title text features.
    for d in range(dim):
        features.add(f'title-dim-{d}', group='title-embedding')


def add_edit_embedding(vec, datum):
    # Edit text features.
    for d, emb in enumerate(datum['edit-embedding']):
        vec[f'edit-dim-{d}'] = emb


def add_title_embedding(vec, datum):
    # Title text features.
    for d, emb in enumerate(datum['title-embedding']):
        vec[f'title-dim-{d}'] = emb


def summarize_features(features):
    print(f'There are {len(features)} features:')
    for group in features.groups():
        print(f'  - {len(features.get_group(group))} {group}')


def load_dataset(path):
    canonical = os.path.abspath(path)
    print(f'Loading canonical dataset {canonical}...')
    return Dataset.load_json(canonical)


def _filter_dossiers(dataset, thr):
    # Count occurence of each dossiers.
    dossiers = list()
    for data in dataset:
        for datum in data:
            dossiers.append(datum['dossier_ref'])
    counter = Counter(dossiers)
    # Define list of dossiers to keep.
    keep = set([d for d, c in counter.items() if c > thr])
    k, d = len(keep), len(set(dossiers))
    print(f'Removed {d-k} ({(d-k)/d*100:.2f}%) dossiers.')
    return keep


def _filter_meps(dataset, thr):
    # Count occurence of each dossiers.
    meps = list()
    for data in dataset:
        for datum in data:
            for at in datum['authors']:
                meps.append(at['id'])
    counter = Counter(meps)
    # Define list of dossiers to keep.
    keep = set([d for d, c in counter.items() if c > thr])
    k, m = len(keep), len(set(meps))
    print(f'Removed {m-k} ({(m-k)/m*100:.2f}%) MEPs.')
    return keep


def filter_dataset(dataset, thr=10):
    """Remove dossiers with less than `thr` edits."""
    keep_doss = _filter_dossiers(dataset, thr)
    keep_mep = _filter_meps(dataset, thr)
    filtered_dataset = list()
    for data in dataset:
        kd, km = True, True
        for datum in data:
            if datum['dossier_ref'] not in keep_doss:
                kd = False
            if not all(at['id'] in keep_mep for at in datum['authors']):
                km = False
        if kd and km:
            filtered_dataset.append(data)
    d, f = len(dataset), len(filtered_dataset)
    print(f'Removed {d-f} ({(d-f)/d*100:.2f}%) conflicts.')
    print('Number of data points:', len(filtered_dataset))
    return filtered_dataset


def _shuffle(featmats, labels, seed):
    np.random.seed(seed)
    perm = np.random.permutation(range(len(featmats)))
    return [featmats[p] for p in perm], [labels[p] for p in perm]


def _split(array, split):
    # Split training and validation sets.
    try:
        idx = int(split)
    except ValueError:  # Split is not an int, maybe a float?
        idx = int(np.ceil(len(array) * float(split)))
    return array[:idx], array[idx:]


def _split_indices(array, indices):
    def get_array(kind):
        return [array[idx] for idx in np.loadtxt(indices[kind], dtype=int)]

    return get_array('train'), get_array('valid'), get_array('test')


def save(feat, featmat, labels, output_path, kind=None):
    # Add a "kind" to the path of the dataset, e.g., "train" of "test".
    if kind is not None:
        path = output_path.replace('.', '-' + kind + '.')
    path = os.path.abspath(path)
    with open(path, 'wb') as f:
        pickle.dump(
            {'features': feat, 'feature_matrices': featmat, 'labels': labels},
            f,
        )
    print(f'Saved to {path}.')


def shuffle_split_save(
    features, featmats, labels, seed, split, output_path, indices=None
):

    # If the split indices are given, preprocess the data using them.
    if indices is not None:
        print('Splitting by pre-defined indices...')
        fmtrain, fmvalid, fmtest = _split_indices(featmats, indices)
        lbtrain, lbvalid, lbtest = _split_indices(labels, indices)
        # Shuffle each split separately.
        print(f'Splits shuffled with seed = {seed}.')
        fmtrain, lbtrain = _shuffle(fmtrain, lbtrain, seed)
        fmvalid, lbvalid = _shuffle(fmvalid, lbvalid, seed)
        fmtest, lbtest = _shuffle(fmtest, lbtest, seed)
        # Save data.
        save(features, fmtrain, lbtrain, output_path, kind='train')
        save(features, fmvalid, lbvalid, output_path, kind='valid')
        save(features, fmtest, lbtest, output_path, kind='test')
        print(f'  Training set: {len(lbtrain)} data points')
        print(f'  Validation set: {len(lbvalid)} data points')
        print(f'  Test set:     {len(lbtest)} data points')

    # Otherwise, use the given seed and split parameters.
    else:
        if seed is not None:
            # Shuffle data.
            featmats, labels = _shuffle(featmats, labels, seed)
            print(f'Dataset shuffled with seed = {seed}.')

        if split is not None:
            # Split data.
            fmtrain, fmtest = _split(featmats, split)
            lbtrain, lbtest = _split(labels, split)
            # Save data.
            save(features, fmtrain, lbtrain, output_path, kind='train')
            save(features, fmtest, lbtest, output_path, kind='test')
            if type(split) is float:
                print(
                    f'Dataset split as {split*100:.0f}%-{(1-split)*100:.0f}%:'
                )
            print(f'  Training set: {len(lbtrain)} data points')
            print(f'  Test set:     {len(lbtest)} data points')
        else:
            # Save data.
            save(features, featmats, labels, output_path)


def get_indices(args):
    if (
        args.train_indices is not None
        and args.valid_indices is not None
        and args.test_indices is not None
    ):
        return {
            'train': args.train_indices,
            'valid': args.valid_indices,
            'test': args.test_indices,
        }
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('canonical', help='Path to canonical dataset')
    parser.add_argument('output_path', help='Path to transformed dataset(s)')
    parser.add_argument(
        '--text-features',
        action='store_true',
        help='Whether to include text features',
    )
    parser.add_argument(
        '--threshold',
        default=None,
        type=int,
        help='Filter dossier with less than threshold edits',
    )
    parser.add_argument(
        '--split', default=None, help='Split between training and test set'
    )
    parser.add_argument(
        '--seed', default=None, type=int, help='Seed for random generator'
    )
    parser.add_argument(
        '--train-indices', default=None, help='Path to indices for train set'
    )
    parser.add_argument(
        '--valid-indices', default=None, help='Path to indices for valid set'
    )
    parser.add_argument(
        '--test-indices', default=None, help='Path to indices for test set'
    )
    return parser.parse_args()
