import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warofwords
from warofwords.utils import build_name, get_base_dir, parse_definition


def log_loss_by_size(data, model):
    size2losses = defaultdict(list)
    size2count = defaultdict(int)
    for X, y in data:
        prob = model.probabilities(X)
        loss = -np.log(prob[y])
        size2losses[len(X)].append(loss)
        size2count[len(X)] += 1
    # print(size2count)
    # Compute log loss.
    size2loss = dict()
    for size, losses in size2losses.items():
        size2loss[size] = np.mean(losses)
    return size2loss


def evaluate(
    definition, data_path, models_path, i, show_fig, save_fig, conflict_size
):
    # Skip definition if 'run' exists and is set to to False.
    if not definition.get('run', True):
        return

    # # Define paths to data, models, and hyperparameters.
    # base_dir = get_base_dir(__file__)
    # data_path = f'{base_dir}/0-datasets/pkl'
    # models_path = f'{base_dir}/trained-models'

    # Extract experiment settings.
    leg, xplct, text, latent, by_dossier, baseline = parse_definition(
        definition
    )
    # Build the name of experiment from the experiment settings.
    name = build_name(leg, xplct, text, latent, by_dossier, baseline)

    # Set model.
    if baseline is not None:
        if 'random' in baseline:
            model = 'Random'
        elif 'naive' in baseline:
            model = 'Naive'
        else:
            raise ValueError(f'Invalid baseline "{baseline}"')
    else:
        model = 'WarOfWordsLatent' if latent else 'WarOfWords'

    # Set values based on experiment settings.
    model_path = f'{models_path}/{name}.predict'
    test_set = f'{data_path}/{name.replace("-latent", "")}-test.pkl'
    dataname = test_set.split('/')[-1].replace('-test.pkl', '')

    # Set model and data.
    TrainedModel = getattr(warofwords, 'Trained' + model)
    trained = TrainedModel.load(model_path)
    _, featmats, labels = trained.load_data(test_set)
    test = list(zip(featmats, labels))
    print(f'{i+1}: Evaluating {model} on {dataname}')

    # Evaluate by conflict size.
    if conflict_size:
        size2loss = log_loss_by_size(test, trained)
        sort = sorted(size2loss.items(), key=lambda k: k[0])
        print('  Conflict size:')
        for size, loss in sort:
            print(f'    {size:>2}: {loss:.4f}')

    # Evaluate.
    acc = trained.accuracy(test)
    los = trained.log_loss(test)
    print(f'  Accuracy: {acc*100:.2f}%')
    print(f'  Log-loss: {los:.4f}')
    if show_fig or save_fig is not None:
        # Plot bar.
        label = '-'.join([model, dataname])
        plt.bar(i + 1, los, color='C0', label=label)

    if conflict_size:
        return {
            'model': model,
            'data': dataname,
            'accuracy': acc,
            'log-loss': los,
            'size2loss': size2loss,
        }
    return {'model': model, 'data': dataname, 'accuracy': acc, 'log-loss': los}


def main(args):
    # Load definition of experiments.
    with open(args.definition) as f:
        definition = json.load(f)

    data = list()
    for i, df in enumerate(definition):
        res = evaluate(
            df,
            args.data_dir,
            args.models_dir,
            i,
            args.show_fig,
            args.save_fig,
            args.conflict_size,
        )
        if args.save_results is not None:
            data.append(res)

    if args.show_fig or args.save_fig is not None:
        # Plot settings.
        plt.grid(axis='y')
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(loc='lower center')

    if args.save_results is not None:
        name = Path(args.definition).name
        path = Path(args.save_results) / name
        with open(path, 'w') as f:
            [f.write(json.dumps(res) + '\n') for res in data]

    if args.save_fig is not None:
        plt.savefig(args.save_fig)
    if args.show_fig:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--definition', help='Path to definition of evaluation'
    )
    parser.add_argument('--data_dir', help='Path to directory of data')
    parser.add_argument('--models_dir', help='Path to save trained models')
    parser.add_argument('--show_fig', action='store_true', help='Show figure')
    parser.add_argument('--save_fig', default=None, help='Path to figure')
    parser.add_argument('--save_results', default=None, help='Path to results')
    parser.add_argument(
        '--conflict_size',
        action='store_true',
        help='Compute loss by conflict size',
    )
    main(parser.parse_args())
