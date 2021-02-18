import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import warofwords.plotting as parlplt

PATH = '{folder}/ep{leg}-{exp}.json'


def get_key(model, data, leg=None):
    if leg is None:
        return (model, data)
    else:
        return (model, f'ep{leg}-{data}')


def get_data(folder, leg, exp):
    res = dict()
    path = PATH.format(folder=folder, leg=leg, exp=exp)
    with open(path, 'r') as f:
        for ln in f.readlines():
            r = json.loads(ln)
            key = get_key(r['model'], r['data'])
            res[key] = r['log-loss']
    return res


def main(args):
    # Load plot settings.
    parlplt.sigconf_settings()
    # Define experiments.
    legs = [7, 8]
    baseline = 'no_features'
    datasets = [
        'dossier_features',
        'mep_features',
        'edit_features',
        'rapporteur_advantage',
        'all_features',
    ]
    model = 'WarOfWords'

    # Get data.
    results = dict()
    for leg in legs:
        results.update(get_data(folder=args.results, leg=leg, exp='results'))
    # Build bars.
    bars = list()
    for dataset in datasets:
        b = list()
        for leg in legs:
            base = results[get_key(model, baseline, leg)]
            key = get_key(model, dataset, leg)
            r = float(f'{results[key]:.3f}')
            b.append(r - float(f'{base:.3f}'))
        bars.append(b)

    # Set figure size.
    fig, ax = plt.subplots(figsize=(3.5, 1.7))

    # Bar settings.
    width = 0.12
    offset = 0.02
    patterns = ['', '///', '\\\\\\', '---', 'xxx']
    colors = ['white', 'white', 'white', 'white', 'white']
    edgecolors = ['black', 'black', 'black', 'black', 'black']
    labels = [
        r'\textsc{WoW}(\em{D})',
        r'\textsc{WoW}(\em{M})',
        r'\textsc{WoW}(\em{E})',
        r'\textsc{WoW}(\em{R})',
        r'\textsc{WoW}(\em{X})',
    ]
    # Get x positions.
    x0 = np.arange(len(bars[0]))
    xs = [
        [x * 0.75 + i * width + i * offset for x in x0]
        for i in range(len(bars))
    ]
    # Draw bars.
    lines = list()
    for i, (x, bar) in enumerate(zip(xs, bars)):
        line = plt.bar(
            x,
            bar,
            width=width,
            color=colors[i],
            linewidth=1.0,
            edgecolor=edgecolors[i],
            hatch=patterns[i],
            label=labels[i],
        )
        lines.append(line)
    # Draw text.
    for bars in lines:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + width / 2,
                height - 0.001,
                f'{height:.3f}',
                ha='center',
                va='top',
                rotation=0,
                fontsize=7,
            )
    # Add xticks on the middle of the group bars
    plt.xticks(
        [x * 0.75 + 1.5 * width + 1 * offset for x in range(len(x0))],
        ['EP7', 'EP8'],
    )
    # Set yticks.
    rng = -np.arange(
        0,
        0.06,
        0.01,
    )
    plt.yticks(rng, ['0.00'] + list(rng[1:]))
    plt.ylim([-0.045, 0.0])
    # Title and legend.
    # plt.title(r'Difference in cross entropy loss over \textsc{WoW}($\cdot$)')
    plt.ylabel(r'Diff. in cross entropy loss')
    plt.legend(
        lines,
        labels,
        loc='lower left',
        frameon=True,
        fontsize='x-small',
        framealpha=0.8,
        markerscale=0.1,
    )
    plt.tight_layout()
    path = os.path.abspath(args.save_as)
    plt.savefig(path, bbox_inches='tight')
    print(f'Figure saved to {path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Path to results')
    parser.add_argument('--save-as', help='Path to figure')
    main(parser.parse_args())
