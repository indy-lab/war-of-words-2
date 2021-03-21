import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from warofwords.plotting import sigconf_settings

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
    sigconf_settings()
    # Define experiments.
    legs = [7, 8]
    models = [
        ('Naive', 'naive'),
        ('Random', 'random'),
        ('WarOfWords', 'no_features'),
        ('WarOfWords', 'all_features'),
        ('WarOfWords', 'no_features-text'),
        ('WarOfWordsLatent', 'no_features'),
        ('WarOfWords', 'all_features-text'),
        ('WarOfWordsLatent', 'all_features'),
        ('WarOfWordsLatent', 'no_features-text'),
        ('WarOfWordsLatent', 'all_features-text'),
    ]
    width = 0.75

    # Get data.
    results = dict()
    for leg in legs:
        results.update(get_data(args.results, leg=leg, exp='results'))
    # Build bars.
    bars = list()
    for leg in legs:
        b = list()
        for model in models:
            b.append(results[get_key(*model, leg)])
        bars.append(b)

    xs = np.arange(len(bars[0]))
    # Set figure size.
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3.5, 3.5))
    fig.subplots_adjust(hspace=0.1)
    # Bar settings.
    patterns = [
        None,
        None,
        None,
        '///',
        '---',
        '\\\\\\',
        '///---',
        'xxx',
        '\\\\\\---',
        '---xxx',
    ]
    colors = [
        'gray',
        'lightgray',
        'white',
        'white',
        'white',
        'white',
        'white',
        'white',
        'white',
        'white',
    ]
    edgecolors = [
        'black',
        'black',
        'black',
        'black',
        'black',
        'black',
        'black',
        'black',
        'black',
        'C3',
    ]
    labels = [
        'Naive',
        'Random',
        r'\textsc{WoW}',
        r'\textsc{WoW}(\em{X})',
        r'\textsc{WoW}(\em{T})',
        r'\textsc{WoW}(\em{L}$)',
        r'\textsc{WoW}(\em{XT})',
        r'\textsc{WoW}(\em{XL})',
        r'\textsc{WoW}(\em{LT})',
        r'\textsc{WoW}(\em{XLT})',
    ]
    positions = ['top', 'bottom']
    titles = [
        r'7\textsuperscript{th} Legislature',
        r'8\textsuperscript{th} Legislature',
    ]
    # Draw bars.
    for ax, legbars, pos, title in zip(axes, bars, positions, titles):
        lines = list()
        for i, (x, bar) in enumerate(zip(xs, legbars)):
            line = ax.bar(
                x,
                bar,
                width=width,
                color=colors[i],
                linewidth=1,
                edgecolor=edgecolors[i],
                hatch=patterns[i],
                label=labels[i],
            )
            lines.append(line)
        # Draw text.
        for b, bar in enumerate(ax.patches):
            height = bar.get_height()
            if '8' in title and b == 0:
                offset = -0.07
                color = 'white'
                size = 6
            else:
                offset = 0.025
                color = 'black'
                size = 6
            ax.text(
                bar.get_x() + width / 2,
                height + offset,
                f'{height:.3f}',
                color=color,
                ha='center',
                va='bottom',
                rotation=0,
                fontsize=size,
            )
        # Add xticks labels.
        if pos == 'top':
            ax.set_xticks([])
            ax.set_xticklabels([])
        elif pos == 'bottom':
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=45)

        # Add title.
        ax.set_title(title)

        ax.set_ylim([0.0, 1.0])
        ax.set_ylabel('Avg. cross entropy')
    path = os.path.abspath(args.save_as)
    plt.savefig(path)
    print(f'Figure saved to {path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Path to results')
    parser.add_argument('--save-as', help='Path to figure')
    main(parser.parse_args())
