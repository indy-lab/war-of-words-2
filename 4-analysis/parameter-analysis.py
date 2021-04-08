import json

from warofwords import TrainedWarOfWordsLatent

LEG = 8


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


# Load mapping from dossier to title.
titles = load_json('../data/helpers/dossier-titles.json')
# Load mapping from dossier to title.
meps = load_json('../data/helpers/meps.json')

# Load trained model to get parameters and features.
models = '../2-training/trained-models'
model = f'{models}/ep{LEG}-all_features-latent-text.fit'
model = TrainedWarOfWordsLatent.load(model)
parameters = model.parameters
features = model.features

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
dossiers = sort_params('dossier', parameters, features, reverse=False, n=10)
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
    grp = meps[mep_id][f'group-ep{LEG}']
    nationality = meps[mep_id]['nationality']
    print(f'{p:+.2f} | {name:<20} | {nationality:<15} | {grp}')

# Print text embeddings.
group = 'title-embedding'
print('\n###', group.upper(), '\n')
print(parameters.get_group(group))
group = 'edit-embedding'
print('\n###', group.upper(), '\n')
print(parameters.get_group(group))
