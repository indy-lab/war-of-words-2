import argparse
import json
from collections import Counter

import numpy as np
from warofwords import Dataset


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


def filter_dataset(dataset, thr):
    """Remove dossiers with less than `thr` edits."""
    print(f'Filtering dataset with threshold = {thr}')
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


def sort_by_date(dataset):
    # Extract start and end dates for each dossier.
    dossier2dates = dict()
    for conflict in dataset:
        dossier = conflict[0]['dossier_ref']
        if dossier not in dossier2dates:
            dossier2dates[dossier] = {
                'num-conflicts': 0,
                'start': None,
                'end': None,
            }
        dossier2dates[dossier]['num-conflicts'] += 1
        for edit in conflict:
            date = edit['date']
            # Set start date.
            start = dossier2dates[dossier]['start']
            if start is None or date < start:
                dossier2dates[dossier]['start'] = date
            # Set end date.
            end = dossier2dates[dossier]['end']
            if end is None or date > end:
                dossier2dates[dossier]['end'] = date

    # Sort dossier by start date.
    ordered_start = sorted(dossier2dates.items(), key=lambda d: d[1]['start'])
    return [(d, v['num-conflicts']) for d, v in ordered_start]


def shuffle(dataset, seed):
    # Extract occurences of dossiers.
    dossiers = list()
    for conflict in dataset:
        dossier = conflict[0]['dossier_ref']
        dossiers.append(dossier)
    # Count number of conflicts per dossier.
    dossier2count = list(Counter(dossiers).items())
    # Shuffle the dossiers and counts.
    np.random.seed(seed)
    np.random.shuffle(dossier2count)

    return dossier2count


def split_by_dossier(dataset, order, seed=100, thr_train=0.8, thr_valid=0.9):
    """Splits the dataset into training and test set, according to dossiers.

    This means that dossiers in the test set are not seen in the training set.

    Arguments:
      - dataset: list of conflicts
      - thr_train: the threshold at which to split the dataset for training
      - thr_valid: the threshold at which to split the dataset for validation
    """

    if order == 'chronological':
        dossier2count = sort_by_date(dataset)
    elif order == 'random':
        dossier2count = shuffle(dataset, seed)
    else:
        raise ValueError(f'Unrecognized order "{order}"')

    # Get total number of conflicts.
    num_conflicts = sum(c for _, c in dossier2count)

    # Assign dossiers to training, validation, or test set.
    train, valid, cumsum = set(), set(), 0
    for i, (dossier, count) in enumerate(dossier2count):
        cumsum += count
        if cumsum / num_conflicts < thr_train:
            train.add(dossier)
        elif thr_train <= cumsum / num_conflicts < thr_valid:
            valid.add(dossier)
        else:
            break
    test = [dossier for dossier, _ in dossier2count[i:]]

    # Generate training, validation, and test set.
    trainset, validset, testset = list(), list(), list()
    for conflict in dataset:
        dossier = conflict[0]['dossier_ref']
        if dossier in train:
            trainset.append(conflict)
        elif dossier in valid:
            validset.append(conflict)
        elif dossier in test:
            testset.append(conflict)
        else:
            raise ValueError(f'Dossier {dossier} not found')

    print('Keep these indices somewhere!')
    print('  Training/validation split:', len(trainset))
    print('  Training/test split:      ', len(trainset) + len(validset))

    # We combine the training, validation, and set back into one datset. The
    # split into training and test set is done in 0-datasets. The split into
    # training and validation is done during 1-grid-search.
    return trainset + validset + testset


def main(args):
    print(f'Loading canonical dataset {args.canonical}...')
    dataset = Dataset.load_json(args.canonical)
    # Filter dossiers and MEPs.
    if args.threshold is not None:
        dataset = filter_dataset(dataset, args.threshold)
    # Separate the dataset by dossier.
    dataset = split_by_dossier(
        dataset, args.order, args.seed, args.split_train, args.split_valid
    )
    # Save new canonical dataset.
    with open(args.output_path, 'w') as f:
        for conflict in dataset:
            # Open JSON list.
            f.write(json.dumps(conflict))
            f.write('\n')
    print(f'Dataset split by dossier saved as {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--canonical', help='Path to canonical dataset')
    parser.add_argument('--output_path', help='Path to new canonical dataset')
    parser.add_argument(
        '--order',
        choices=['chronological', 'random'],
        help='Choose how to order the dossiers',
    )
    parser.add_argument(
        '--seed', default=100, type=int, help='Seed used when order=random'
    )
    parser.add_argument(
        '--threshold',
        default=None,
        type=int,
        help='Filter dossier with less than threshold edits',
    )
    parser.add_argument(
        '--split_train',
        default=0.8,
        type=float,
        help='Split between training and validation sets',
    )
    parser.add_argument(
        '--split_valid',
        default=0.9,
        type=float,
        help='Split between validation and test sets',
    )
    main(parser.parse_args())
