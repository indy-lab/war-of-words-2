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


def shuffle_split_save(features, featmats, labels, seed, split, output_path):
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
            print(f'Dataset split as {split*100:.0f}%-{(1-split)*100:.0f}%:')
        print(f'  Training set: {len(lbtrain)} data points')
        print(f'  Test set:     {len(lbtest)} data points')
    else:
        # Save data.
        save(features, featmats, labels, output_path)


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
    return parser.parse_args()
