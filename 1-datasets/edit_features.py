import os

import numpy as np
from warofwords import Features

from _common import (filter_dataset, load_dataset, parse_args,
                     shuffle_split_save, summarize_features)


def main(args):
    # Load canonical dataset.
    dataset = load_dataset(os.path.abspath(args.canonical))

    if args.threshold is not None:
        print('Filtering dataset...')
        dataset = filter_dataset(dataset, thr=args.threshold)

    # Initialize features.
    features = Features()
    features.add('bias', group='bias')
    features.add('insert-length', group='edit-length')
    features.add('delete-length', group='edit-length')
    features.add('justification', group='justification')
    features.add('outsider', group='outsider')
    for data in dataset:
        for datum in data:
            # Edit features.
            features.add(datum['dossier_ref'], group='dossier')
            features.add(datum['article_type'], group='article-type')
            features.add(datum['edit_type'], group='edit-type')
            # Add the skills.
            for a in datum['authors']:
                features.add(a['id'], group='mep')

    # Print summary of the features.
    summarize_features(features)

    # Build feature matrices and extract labels.
    print('Transforming dataset...')
    # Each data point is a feature matrix of N_k features, where N_k is the
    # total number of features (number of conflicting edits+1 for the dossier).
    featmats = list()
    labels = list()
    for data in dataset:
        featmat = list()
        label = None

        # Extract labels and MEP/edit features.
        for i, datum in enumerate(data):
            vec = features.new_vector()
            # Add article type.
            vec[datum['article_type']] = 1
            # Add edit type.
            vec[datum['edit_type']] = 1
            # Add outisder advantage (whether the edit is proposed by another
            # committee).
            if datum['outsider']:
                vec['outsider'] = 1
            # Add justification.
            just = datum['justification']
            # With text features, the justification is either None or the whole
            # text, without text it is True or False.
            if type(just) is not bool:
                just = (just is not None) and (just != '')
            if just:
                vec['justification'] = 1
            # Add edit length.
            i1, i2 = datum['edit_indices']['i1'], datum['edit_indices']['i2']
            j1, j2 = datum['edit_indices']['j1'], datum['edit_indices']['j2']
            vec['delete-length'] = np.log(1 + i2 - i1)
            vec['insert-length'] = np.log(1 + j2 - j1)
            # Add MEP features.
            for a in datum['authors']:
                vec[a['id']] = 1
            featmat.append(vec.as_sparse_list())
            # Add label if edit is accepted.
            if datum['accepted']:
                label = i

        # Add dossier features.
        vec = features.new_vector()
        vec[data[0]['dossier_ref']] = 1
        vec['bias'] = 1
        featmat.append(vec.as_sparse_list())

        # Add label if dossier wins conflict.
        if label is None:
            label = len(data)  # Set to the last one.
        labels.append(label)

        # Add feature matrix.
        featmats.append(featmat)

    shuffle_split_save(
        features, featmats, labels, args.seed, args.split, args.output_path
    )


if __name__ == "__main__":
    main(parse_args())
