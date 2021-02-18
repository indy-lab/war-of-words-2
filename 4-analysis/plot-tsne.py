import matplotlib.pyplot as plt
from parlpapr.plotting import sigconf_settings


colors = {
    'ds': 'C3',
    'ff': 'C1',
    'ss': 'C2',
    'cl': 'C0',
    'ot': 'lightgray',
}

markers = {
    'ds': 'X',
    'ff': 's',
    'ss': 'v',
    'cl': 'o',
    'ot': '.',
}

labels = [
    r'Defense \& Support',
    r'Freedom \& Fairness',
    r'Safety \& Solidarity',
    r'Citizen \& Lifestyle',
    r'Others'
]

# %%
sigconf_settings()
fig, ax = plt.subplots(figsize=(3.6, 1.7))

lines = dict()

for i, (x, y) in enumerate(embed):
    ref = doss[i]
    cluster = ref2cluster[ref]
    line = ax.plot(x, y, color=colors[cluster], marker=markers[cluster])
    if cluster not in line:
        lines[cluster] = line[0]

handles = [lines['ds'],
           lines['ff'],
           lines['ss'],
           lines['cl'],
           lines['ot']]


plt.legend(handles, labels,
           loc='upper left',
           frameon=True,
           fontsize='x-small',
           bbox_to_anchor=(0.60, 0.80))
plt.tick_params(
    axis='both',        # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    left=False,         # tickcs on the left edge are off
    top=False,          # ticks along the top edge are off
    labelbottom=False,  # labels along the bottom edge are off
    labelleft=False)    # labels along the bottom edge are off
# plt.title('t-SNE visualization of latent feature y_i')
plt.tight_layout()
plt.savefig('../paper/fig/tsne.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
"""
### Generate t-SNE plots for visual inspection
"""

# %%
ps = [4, 4.5, 5]
ss = list(range(0, 20))
for p, s in product(ps, ss):
    print(p, s)
    embed, coords, _ = plot_tsne(pca, vec_y, perplexity=p, seed=s, n=10)




def get_dossiers(features, vec):
    """Get top dossiers and their latent features."""
    doss = features.get_group('dossier')
    vec = vec[doss]
    return [features.get_name(d) for d in doss], vec.numpy()


def main():
    # Load mapping to titles.
    dossier2title = dict()
    with open(f'{BASE}/scrp/data/ep{LEG}-dossier2title.json', 'r') as f:
        for l in f.readlines():
            d = json.loads(l)
            for k, v in d.items():
                dossier2title[k.replace('_', '-')] = v

    # Get all dossiers and their latent features.
    doss, vec_y = get_dossiers(features, vec)
