import argparse
import json
import re

import numpy as np
from warofwords import TrainedWarOfWordsLatent


def sort_params(group, parameters, features, reverse=True, n=None):
    params = [
        (feat, parameters[feat])
        for feat in features.get_group(group, by_feature_name=True)
    ]
    if n is None:
        n = len(params)
    return sorted(params, key=lambda p: p[1], reverse=reverse)[:n]


def print_params(group, parameters, features, n=None):
    print('###', group.upper(), '\n')
    sortedparams = sort_params(group, parameters, features, n=n)
    for name, p in sortedparams:
        print(f'{p:+.2f} {name}')
    print()


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def main(args):
    # Load mapping from dossier to title.
    titles = load_json(args.dossier_titles)
    # Load mapping from dossier to title.
    meps = load_json(args.meps)

    # Load trained model to get parameters and features.
    model = TrainedWarOfWordsLatent.load(args.model)
    parameters = model.parameters
    features = model.features

    # Extract legislature from model path.
    m = re.search(r'ep(\d)', args.model)
    if m:
        leg = m.group(1)
    else:
        raise ValueError(
            f'Model path "{args.model}" has no reference to legislature period'
        )

    # Define groups of features to analyze.
    groups = [
        # MEP features.
        'political-group',
        'nationality',
        'gender',
        'rapporteur',
        # Edit features.
        'edit-length',
        'justification',
        'outsider',
        'article-type',
        'edit-type',
        # Dossier features.
        'dossier-type',
        'legal-act',
        'committee',
    ]
    for group in groups:
        print_params(group, parameters, features)

    # Print top-10 and bottom-10 dossiers.
    print('### TOP-10 DOSSIERS\n')
    header = f'PARAM | {"DOSSIER":<19} | TITLE'
    print(header)
    print('-' * len(header))
    dossiers = sort_params('dossier', parameters, features, reverse=True, n=10)
    for doss, p in dossiers:
        title = titles[doss]
        print(f'{p:+.2f} | {doss:<19} | {title}')
    print('\n### BOTTOM-10 DOSSIERS\n')
    dossiers = sort_params(
        'dossier', parameters, features, reverse=False, n=10
    )
    for doss, p in dossiers:
        title = titles[doss]
        print(f'{p:+.2f} | {doss:<19} | {title}')

    # Print top-10 MEPs.
    print('\n### MEPS\n')
    mep_ids = sort_params('mep', parameters, features, n=10)
    header = f'PARAM | {"NAME":<20} | {"NATIONALITY":<15} | POLITICAL GROUP'
    print(header)
    print('-' * len(header))
    for mep_id, p in mep_ids:
        mep_id = str(mep_id)
        name = meps[mep_id]['name']
        grp = meps[mep_id][f'group-ep{leg}']
        nationality = meps[mep_id]['nationality']
        print(f'{p:+.2f} | {name:<20} | {nationality:<15} | {grp}')

    # Print text embeddings.
    group = 'title-embedding'
    print('\n###', group.upper(), '\n')
    print(parameters.get_group(group))
    if args.save_text_embeddings:
        np.savetxt('title-parameters.txt', parameters.get_group(group))
    group = 'edit-embedding'
    print('\n###', group.upper(), '\n')
    print(parameters.get_group(group))
    if args.save_text_embeddings:
        np.savetxt('edit-parameters.txt', parameters.get_group(group))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to model to analyze')
    parser.add_argument(
        '--dossier-titles',
        dest='dossier_titles',
        help='Path to mapping of dossier titles',
    )
    parser.add_argument('--meps', help='Path to MEPs metadata')
    parser.add_argument(
        '--save-text-embeddings',
        action='store_true',
        help='Whether to save the text embeddings as text file',
    )
    main(parser.parse_args())
