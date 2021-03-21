import json

# from parldata import Parliamentarian
from warofwords import TrainedWarOfWordsLatent

LEG = 8

# Load mapping from dossier to title.
with open('../data/helpers/dossier-titles.json', 'r') as f:
    titles = json.load(f)


def sort_params(group, parameters, features, reverse=True, n=10):
    params = [
        (feat, parameters[feat])
        for feat in features.get_group(group, by_feature_name=True)
    ]
    return sorted(params, key=lambda p: p[1], reverse=reverse)[:n]


models = '../2-training/trained-models'
model = f'{models}/ep{LEG}-all_features-latent-text.fit'
model = TrainedWarOfWordsLatent.load(model)

parameters = model.parameters
features = model.features

groups = features.groups()
groups = [
    'rapporteur',
    'edit-length',
    'justification',
    'outsider',
    'article-type',
    'edit-type',
    'dossier-type',
    'legal-act',
    'gender',
]


for group in groups:
    print(group.upper())
    for feat, p in sort_params(group, parameters, features):
        print(f'{p:+.2f} {feat}')

print('TOP-10 DOSSIERS')
dossiers = sort_params('dossier', parameters, features, reverse=True, n=10)
for doss, p in dossiers:
    title = titles[doss]
    print(f'{p:+.2f} {doss} {title}')

print('BOTTOM-10 DOSSIERS')
dossiers = sort_params('dossier', parameters, features, reverse=False, n=50)
for doss, p in dossiers:
    title = titles[doss]
    print(f'{p:+.2f} {doss} {title}')

# print('MEPS')
# meps = sort_params('mep', parameters, features, n=10)
# for mep, p in meps:
#     name = (
#         Parliamentarian.select(Parliamentarian.displayname)
#         .where(Parliamentarian.europarl_id == mep)
#         .get()
#         .displayname
#     )
#     print(f'{p:+.2f} {name}')


group = 'political-group'
print(group.upper())
polgroups = sort_params(group, parameters, features, n=10)
for grp, p in polgroups:
    print(f'{p:+.2f} {grp}')


group = 'nationality'
print(group.upper())
nationalities = sort_params(group, parameters, features, n=28)
for nat, p in nationalities:
    print(f'{p:+.2f} {nat}')

group = 'title-embedding'
print(group.upper())
print(parameters.get_group(group))

group = 'edit-embedding'
print(group.upper())
print(parameters.get_group(group))
