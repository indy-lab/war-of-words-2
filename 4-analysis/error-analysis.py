import argparse

import matplotlib.pyplot as plt
import warofwords.plotting as parlplt

counts = {
    # Bin.
    2: 8462,
    # Bin.
    3: 3063,
    # Bin.
    4: 1380,
    # Bin.
    5: 621,
    # Bin.
    6: 288,
    # Bin.
    7: 125,
    # Bin.
    8: 57,
    9: 30,
    10: 15,
    11: 14,
    12: 10,
    13: 8,
    14: 5,
    15: 1,
}

random = {
    2: 0.6924945914040918,
    3: 0.9471702655961095,
    7: 1.137898902331221,
    4: 1.010772422799787,
    5: 1.0475588776506537,
    6: 1.173686840176666,
    8: 1.2962453935378917,
    13: 3.169177387754107,
    11: 0.5165646775437585,
    14: 2.707274423916508,
    9: 1.033282042559301,
    10: 0.7799774660462623,
    12: 2.1108114535189078,
    15: 0.7621400520468967,
}

no_features = {
    2: 0.6280587204107826,
    3: 0.8389253029917922,
    7: 1.3825555809342556,
    4: 0.9946506787960245,
    5: 1.2285265266523984,
    6: 1.3823651777163661,
    8: 1.349290718781149,
    13: 2.2320323050115674,
    11: 1.5867149538255554,
    14: 2.4490526955374494,
    9: 1.2006417096171196,
    10: 1.1257084255871335,
    12: 1.670502105856713,
    15: 1.1276498074853438,
}

wowxlt = {
    2: 0.5244311314520334,
    3: 0.6924398063825777,
    7: 1.0649597703877363,
    4: 0.7976585441273126,
    5: 1.0170018551849025,
    6: 1.082789269650528,
    8: 1.0491800700696707,
    13: 2.3525607023557242,
    11: 1.0497601760608244,
    14: 2.272332951599246,
    9: 0.9201263260187946,
    10: 0.7385799997133209,
    12: 1.4944354455645612,
    15: 0.7898231723180078,
}


def generate_series(losses, counts, bins):
    series = list()
    for bn in bins:
        total = sum(counts.get(b, 0) for b in bn)
        print(bn, total)
        loss = sum(counts.get(b, 0) * losses[b] for b in bn) / total
        series.append(loss)
    return series


def main(args):
    bins = [
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8, 9, 10, 11, 12, 13, 14, 15],
    ]

    labels = ['Random', r'\textsc{WoW}', r'\textsc{WoW}(\em{XLT})']
    colors = ['black', 'black', 'C3']

    lines = [':', '--', '-']
    markers = ['o', 'X', '^']

    yrandom = generate_series(random, counts, bins)
    ynofeat = generate_series(no_features, counts, bins)
    ywowxlt = generate_series(wowxlt, counts, bins)

    xs = range(len(bins))
    ys = [yrandom, ynofeat, ywowxlt]

    xtickslabels = ['1', '2', '3', '4', '5', '6', r'7--14']

    parlplt.sigconf_settings()

    # Set figure size.
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for i, y in enumerate(ys):
        ax.plot(
            xs,
            ys[i],
            linestyle=lines[i],
            linewidth=2,
            marker=markers[i],
            color=colors[i],
            label=labels[i],
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(xtickslabels)
    plt.ylabel('Avg. cross entropy')
    plt.xlabel(r'Conflict size $\vert \mathcal{C} \vert = K$')
    plt.legend(
        frameon=True, fontsize='x-small', framealpha=0.8, markerscale=0.1
    )
    plt.tight_layout()
    plt.savefig(args.save_as, bbox_inches='tight')
    print(f'Figure saved to {args.save_as}')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--results', help='Path to JSON of results')
    parser.add_argument('--save-as', help='Path to figure')
    main(parser.parse_args())
