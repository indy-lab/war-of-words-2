import matplotlib.pyplot as plt
import numpy as np
import pickle

from pathlib import Path


def get_base_dir(file):
    filename = Path(file)  # Current script.
    return Path(filename).resolve().parent.parent  # Two levels above.


def parse_definition(definition):
    leg = definition['legislature']
    explicit = definition.get('explicit-features', None)
    text = definition.get('text-features', None)
    latent = definition.get('latent-features', False)
    by_dossier = definition.get('by-dossier', False)
    random = definition.get('random', None)
    return leg, explicit, text, latent, by_dossier, random


def build_name(leg, explicit, text, latent, by_dossier, random):
    name = f'ep{leg}'
    if random is not None:
        name += f'-{random}'
    else:
        if explicit is None:
            raise ValueError('You must specify some explicit features')
        name += f'-{explicit}'
        if latent:
            name += '-latent'
        if text is not None:
            name += f'-{text}'
    if by_dossier:
        name += '-by_dossier'
    return name


def load_pkl(path):
    """Load a pickle from path."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def display_parameters(group, features, params, n=10):
    feats = [(features.get_name(idx), params[features.get_name(idx)])
             for idx in features.get_group(group)]
    for name, param in sorted(feats, key=lambda k: k[1], reverse=True)[:n]:
        if param != 0:
            print(f'{param:.4f} {name}')


def train_save(model, trained_model, hyper, input_path, output_path, verbose):

    # Load dataset.
    features, featmats, labels = model.load_data(input_path)
    train = list(zip(featmats, labels))

    # Initialize model.
    model = model(train, features, hyper, verbose)

    # Train.
    params, cost = model.fit()
    trained = trained_model(features, hyper, **params)
    trained.save(output_path)


def barchart(y7, y8, obj, config, figpath=None):
    width = config['width']
    offset = config['offset']

    bars = [[y7[i], y8[i]] for i in range(len(obj))]

    r0 = np.arange(len(bars[0]))
    rs = [[x*0.75 + i * width + i * offset for x in r0]
          for i in range(len(bars))]

    fig, ax = plt.subplots(figsize=(3.5, 1.9))

    lines = list()
    for i, (r, ys) in enumerate(zip(rs, bars)):
        line = plt.bar(r, ys,
                       width=width,
                       color=config['colors'][i],
                       linewidth=1,
                       edgecolor=config['edgecolors'][i],
                       hatch=config['patterns'][i],
                       label=obj[i])
        lines.append(line)

    # Add xticks on the middle of the group bars
    # plt.xlabel('group', fontweight='bold')
    plt.ylabel(config['ylabel'])
    # plt.xticks([r*0.75 + 2*width + 2*offset for r in range(len(r0))],
    plt.xticks([r*0.75 + width + offset for r in range(len(r0))],
               [r'7\textsuperscript{th} legislature',
                r'8\textsuperscript{th} legislature'])

    plt.ylim([0., 1.])
    plt.legend(
        lines,
        obj,
        loc='lower center',
        frameon=True,
        fontsize='x-small',
        framealpha=0.9,
        markerscale=0.1
    )
    plt.tight_layout()
    if figpath is not None:
        plt.savefig(figpath)
    plt.show()


def get_value(arr):
    mid = len(arr) // 2
    return arr[:mid], arr[mid:]


def k_fold_gen(data, k_fold=10):
    subset_size = int(len(data) / k_fold)
    for k in range(k_fold):
        train = data[:k * subset_size] + data[(k + 1) * subset_size:]
        valid = data[k * subset_size:][:subset_size]

        yield train, valid
