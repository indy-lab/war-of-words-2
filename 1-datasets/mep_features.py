import os
import pickle

from warofwords import Features

from _common import (add_edit_embedding, add_text_features,
                     add_title_embedding, filter_dataset, get_indices,
                     load_dataset, parse_args, shuffle_split_save,
                     summarize_features)


def main(args):
    # Load canonical dataset.
    dataset = load_dataset(os.path.abspath(args.canonical))

    if args.threshold is not None:
        print('Filtering dataset...')
        dataset = filter_dataset(dataset, thr=args.threshold)

    # Initialize features.
    features = Features()
    features.add('bias', group='bias')
    for data in dataset:
        for datum in data:
            features.add(datum['dossier_ref'], group='dossier')
            for a in datum['authors']:
                features.add(a['id'], group='mep')
                features.add(a['nationality'], group='nationality')
                features.add(a['group'], group='political-group')
                features.add(a['gender'], group='gender')

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

        # Extract labels and MEP features.
        for i, datum in enumerate(data):
            vec = features.new_vector()
            # Get MEP ids.
            for a in datum['authors']:
                vec[a['id']] = 1
                vec[a['nationality']] = 1
                vec[a['group']] = 1
                vec[a['gender']] = 1

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
        features,
        featmats,
        labels,
        args.seed,
        args.split,
        args.output_path,
        get_indices(args),
    )


if __name__ == "__main__":
    main(parse_args())
