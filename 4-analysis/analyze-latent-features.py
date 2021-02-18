import json
import matplotlib.pyplot as plt
import numpy as np

from parlpred import TrainedWarOfWordsLatent
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


LEG = 8
EXPLICIT = 'all_features'
BASE = '/Users/kristof/GitHub/parl/pred'
MODELS_PATH = f'{BASE}/trained-models'
DATA_PATH = f'{BASE}/0-datasets/pkl'


def get_embeddings(features, vec, which):
    """Get dossiers and their latent features."""
    if which not in ['mep', 'dossier']:
        raise ValueError(f'Cannot get embeddings for "{which}"')
    doss = features.get_group(which)
    vec = vec[doss]
    return [features.get_name(d) for d in doss], vec


def plot_tsne(pca, vec_y, perplexity, seed=0, n=10):
    """Plots t-SNE embedding of the top-n and bottom-n dossiers of the first
    two principal components of the latent features."""
    coords = pca.transform(vec_y)
    dims = [0, 1]
    # Get dossiers (avoid double counting).
    n = 10
    idx = set()
    for dim in dims:
        # Top-n dossiers.
        idx.update(np.argsort(coords[:, dim])[::-1][:n])
        # Bottom-n dossiers.
        idx.update(np.argsort(coords[:, dim])[:n])
    idx = np.array(list(idx))
    # Compute t-SNE
    tsne = TSNE(perplexity=perplexity, random_state=seed)
    embed = tsne.fit_transform(vec_y[idx, :])
    plt.scatter(embed[:, 0], embed[:, 1])
    plt.show()
    return embed, coords, idx


def display_dossiers(embed, coords, idx, x_thr, y_thr, dossiers, titles):
    for i, (x, y) in zip(idx, embed):
        cond_x = (x > x_thr) if x_thr >= 0 else (x < x_thr)
        cond_y = (y > y_thr) if y_thr >= 0 else (y < y_thr)
        if cond_x and cond_y:
            ref = dossiers[i]
            title = titles[ref]
            print(f'({x:+.0f}, {y:+.0f}) {ref:<10} {title}')


# Load features.
path = f'{DATA_PATH}/ep{LEG}-{EXPLICIT}-train.pkl'
features, _, _ = TrainedWarOfWordsLatent.load_data(path)

# Load model and get latent features of dossiers.
model_path = f'{MODELS_PATH}/ep{LEG}-{EXPLICIT}-latent-text.predict'
model = TrainedWarOfWordsLatent.load(model_path)
vec = model._vec

# Load mapping from dossier to title.
titles = dict()
with open(f'{BASE}/../data/data/helpers/ep{LEG}-dossier2title.json', 'r') as f:
    for ln in f.readlines():
        d = json.loads(ln)
        for k, v in d.items():
            titles[k.replace('_', '-')] = v

# Get dossier latent features.
dossiers, vec_y = get_embeddings(features, vec, 'dossier')

# Apply PCA on the latent features of dossiers.
n_dims = 10
pca = PCA(n_components=n_dims)
pca.fit(vec_y)

# Plot ratio of variance explained.
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(n_dims), pca.explained_variance_ratio_)
ax.set_title("Ratio of variance explained by each PCA dimension")
plt.show()

# Plot embedding with t-SNE.
embed, coords, idx = plot_tsne(pca, vec_y, perplexity=3, seed=5, n=10)

# Display dossiers in each part of the space.
left, bottom = -1e-5, -1e-5
right, top = 1e-5, 1e-5
quandrants = [
    (right, top, 'Quandrant 1'),
    (left, top, 'Quandrant 2'),
    (left, bottom, 'Quandrant 3'),
    (right, bottom, 'Quandrant 4')
]
for x, y, pos in quandrants:
    print(pos)
    display_dossiers(embed, coords, idx, x, y, dossiers, titles)
    print()



# Quandrant 1
(+146, +197) A8-0412-2018 Establishing the European Defence Fund
(+118, +378) A8-0483-2018 Low carbon benchmarks and positive carbon impact benchmarks
(+183, +277) REGI-AD(2018)627807 Protection of the Union's budget in case of generalised deficiencies as regards the ru
le of law in the Member States
(+54, +141) A8-0037-2018 Establishing the European Defence Industrial Development Programme aiming at supporting the co
mpetitiveness and innovative capacity of the EU defence industry
(+168, +246) A8-0064-2017 Amendment to Directive 2004/37/EC on the protection of workers from the risks related to expo
sure to carcinogens or mutagens at work
(+232, +278) A8-0057-2018 Proposal for a Directive of the European Parliament and of the Council to empower the competi
tion authorities of the Member States to be more effective enforcers and to ensure the proper functioning of the intern
al market
(+118, +160) A8-0409-2018 Establishing the Connecting Europe Facility
(+174, +362) A8-0038-2014 Possibility for the Member States to restrict or prohibit the cultivation of GMOs in their te
rritory
(+86, +372) A8-0363-2018 Disclosures relating to sustainable investments and sustainability risks

Quandrant 2
(-71, +246) A8-0295-2018 Prudential supervision of investment firms
(-76, +309) EMPL-AD(2017)601064 Standards for the qualification of third-country nationals or stateless persons as bene
ficiaries of international protection, for a uniform status for refugees or for persons eligible for subsidiary protect
ion and for the content of the protection granted and amending Council Directive 2003/109/EC of 25 November 2003 concer
ning the status of third-country nationals who are long-term residents
(-128, +264) A8-0421-2018 Establishing the "Fiscalis" programme for cooperation in the field of taxation
(-42, +305) CULT-AD(2017)595657 European Electronic Communications Code (Recast)
(-5, +333) A8-0322-2017 European Travel Information and Authorisation System (ETIAS) and amending Regulations (EU) No 5
15/2014, (EU) 2016/399 and (EU) 2016/1624
(-179, +261) A8-0139-2015 European Fund for Strategic Investments
(-89, +174) A8-0482-2018 Establishing the InvestEU Programme
(-56, +213) A8-0011-2019 European Union macro-prudential oversight of the financial system and establishing a European
Systemic Risk Board

Quandrant 3
(-199, -68) A8-0199-2015 Protection of undisclosed know-how and business information (trade secrets) against their unla
wful acquisition, use and disclosure
(-130, -167) AFET-AD(2018)616888 Establishing a framework for screening of foreign direct investments into the European
 Union
(-103, -184) A8-0461-2018 European Social Fund Plus (ESF+)
(-164, -126) A8-0438-2018 Re-use of public sector information (recast)
(-118, -242) A8-0211-2017 Financial rules applicable to the general budget of the Union
(-64, -203) CULT-AD(2018)627580 Establishing the InvestEU Programme

Quandrant 4
(+29, -169) TRAN-AD(2018)623885 Establishing a European Labour Authority
(+204, -465) A8-0238-2016 Prospectus to be published when securities are offered to the public or admitted to trading
(+104, -169) A8-0278-2018 Pan-European Personal Pension Product (PEPP)
(+230, -443) A8-0008-2017 Establishing a Union programme to support specific activities enhancing the involvement of co
nsumers and other financial services end-users in Union policy making in the field of financial services for the period
 of 2017-2020
(+149, -246) A8-0305-2017 Body of European Regulators for Electronic Communications
(+50, -236) A8-0317-2018 Reduction of the impact of certain plastic products on the environment
(+108, -274) AFET-AD(2018)612300 Establishing the European Defence Industrial Development Programme aiming at supportin
g the competitiveness and innovative capacity of the EU defence industry
(+206, -412) CULT-AD(2017)595592 Rules on the exercise of copyright and related rights applicable to certain online tra
nsmissions of broadcasting organisations and retransmissions of television and radio programmes
(+42, -116) A8-0318-2017 European Electronic Communications Code (Recast)
(+102, -326) A8-0198-2017 Extension of the duration of the European Fund for Strategic Investments as well as the intro
duction of technical enhancements for that Fund and the European Investment Advisory Hub
(+181, -343) A8-0011-2016 Activities and supervision of institutions for occupational retirement provision (recast)
(+65, -206) LIBE-AD(2018)620997 Import of cultural goods
