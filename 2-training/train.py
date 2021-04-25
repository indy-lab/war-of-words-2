import argparse
import json
import os

import warofwords
from warofwords.utils import build_name, get_base_dir, parse_definition


def train_save(Model, TrainedModel, hyper, train_set, model_path, verbose):
    """Train and save the model."""
    print(f'Training {Model.__name__} on {train_set}...')
    if hyper is not None:
        print(f'Hyperparameters: {hyper}')
    # Load dataset.
    features, featmats, labels = Model.load_data(train_set)
    train = list(zip(featmats, labels))

    # Initialize model.
    model = Model(train, features, hyper, verbose)

    # Train.
    params, cost = model.fit()
    trained = TrainedModel(features, hyper, **params)
    # Save model.
    trained.save(model_path)
    print(f'Saved to {model_path.split("/")[-1]}.')


def run_experiment(definition, data_path, hyper_path, models_path, verbose):
    # Skip definition if 'run' exists and is set to to False.
    if not definition.get('run', True):
        return

    # Extract experiment settings.
    leg, xplct, text, latent, chronological, baseline, fit = parse_definition(
        definition
    )
    # Build the name of experiment from the experiment settings.
    name = build_name(leg, xplct, text, latent, chronological, baseline)

    # Set values based on experiment settings.
    hp = f'{hyper_path}/{name}.json'
    if fit:
        train_set = f'{data_path}/{name.replace("-latent", "")}-fit.pkl'
        model_path = f'{models_path}/{name}.fit'
    else:
        train_set = f'{data_path}/{name.replace("-latent", "")}-train.pkl'
        model_path = f'{models_path}/{name}.predict'

    if os.path.exists(model_path):
        print(f'Model "{name}" already trained')
        return

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

    # Set model.
    Model = getattr(warofwords, model)
    Trained = getattr(warofwords, 'Trained' + model)
    # Load hyperparameters.
    if baseline is None:
        with open(hp, 'r') as f:
            hyper = json.load(f)
    else:
        hyper = None
    train_save(Model, Trained, hyper, train_set, model_path, verbose)


def main(args):
    # Load experiment config.
    with open(args.definition) as f:
        definition = json.load(f)

    # Run list of experiments.
    if type(definition) is list:
        for df in definition:
            run_experiment(
                df,
                args.data_dir,
                args.hyperparams_dir,
                args.models_dir,
                args.verbose,
            )
    # Run one experiment.
    else:
        run_experiment(
            definition,
            args.data_dir,
            args.hyperparams_dir,
            args.models_dir,
            args.verbose,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--definition', help='Path to definition of experiments'
    )
    parser.add_argument('--data_dir', help='Path to directory of data')
    parser.add_argument('--hyperparams_dir', help='Path to hyperparams')
    parser.add_argument('--models_dir', help='Path to save trained models')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print some information about the training',
    )
    main(parser.parse_args())
