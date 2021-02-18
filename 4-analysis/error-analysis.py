import argparse

import matplotlib.pyplot as plt
import warofwords.plotting as parlplt

counts = {
    2: 8423,
    # Bin.
    3: 3089,
    # Bin.
    4: 1391,
    # Bin.
    5: 594,
    # Bin.
    6: 264,
    # Bin.
    7: 130,
    8: 78,
    # Bin.
    9: 25,
    10: 32,
    11: 16,
    12: 15,
    13: 4,
    14: 5,
    15: 4,
    16: 2,
    17: 1,
    18: 3,
}

random = {
    2: 0.6928513958923126,
    3: 0.9373328026798228,
    4: 1.0259534330100573,
    7: 1.1484030908080076,
    5: 1.0666357802409456,
    15: 1.4149086607915908,
    6: 1.083896214589789,
    12: 1.2234799560471823,
    8: 1.1820283588139626,
    14: 2.2160136999047255,
    9: 0.45999175356632543,
    11: 0.9184108846196273,
    10: 1.122960162042105,
    16: 2.0794415416798357,
    18: 3.285198467799274,
    13: 1.9514190479255367,
    17: 0.5306282510621704,
}

no_features = {
    2: 0.6207271297327831,
    3: 0.8240653398265895,
    4: 0.9646654320698739,
    7: 1.2452285952831559,
    5: 1.0596765251153555,
    15: 1.137549573057839,
    6: 1.0599403033115478,
    12: 1.3187535482382884,
    8: 1.0851528634382694,
    14: 1.39335084834707,
    9: 1.3083163870797474,
    11: 1.5914693436287335,
    10: 1.1271294174032058,
    16: 1.3863202910266668,
    18: 1.394800742035604,
    13: 1.3683912754249667,
    17: 1.2296928290715743,
}


wowxlt = {
    2: 0.5248010422953449,
    3: 0.6475453310604433,
    4: 0.6726144101237896,
    7: 0.7294412133079591,
    5: 0.686146972280061,
    15: 0.22992138490887587,
    6: 0.6903086657919576,
    12: 0.5357824150621018,
    8: 0.6828953202181796,
    14: 0.16030977446764286,
    9: 0.519330947856583,
    11: 0.7245395757074792,
    10: 0.5218512794084622,
    16: 0.18135259938144915,
    18: 0.02818876347484775,
    13: 0.5273533923294111,
    17: 0.36449769375344354,
}


def generate_series(losses, counts, bins):
    series = list()
    for bn in bins:
        total = sum(counts[b] for b in bn)
        print(bn, total)
        loss = sum(counts[b] * losses[b] for b in bn) / total
        series.append(loss)
    return series


def main(args):
    bins = [
        [2],
        [3],
        [4],
        [5],
        [6],
        [7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
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

    xtickslabels = ['1', '2', '3', '4', '5', r'6--7', r'8--17']

    parlplt.sigconf_settings()

    # Set figure size.
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
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
    parser.add_argument('--save-as', help='Path to figure')
    main(parser.parse_args())
