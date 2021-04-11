# War of Words II: Enriched Models of Law-Making Processes

Data and code for

> Victor Kristof, Aswin Suresh, Matthias Grossglauser, Patrick Thiran, [_War of Words II: Enriched Models of Law-Making Processes_](https://infoscience.epfl.ch/record/284828), The Web Conference 2021, April 19-23, 2021, Ljubljana, Slovenia.

**Note:** The repo for _War of Words: The Competitive Dynamics of Legislative Processes_ is [here](https://github.com/indy-lab/war-of-words).

## Set up

From the root of the repo, install the requirements and local library:

```
pip install -r requirements.txt
pip install -e lib
```

## Data

Download the (raw) data from [link]()

Put the canonical datasets (`war-of-words-2-ep{7,8}.txt`) in a folder in the repo, for example:

```
mkdir -p data/canonical
```

If you don't want to generate the text embeddings from scratch, download `ep{7,8}-text-embeddings.txt` and put them in

```
mkdir data/text-embeddings
```

Also download the helper files (a mapping of dossier references to their title and some MEPs metadata) and put them in

```
mkdir data/helpers
```

You should also put the files containing the indices to split the data into train and test sets in 

```
mkdir data/split-indices
```


## Step 0: Generate the text embeddings

You can generate the text embeddings for the 7th and 8th legislatures by running

```
python generate_embeddings.py --leg {7,8} --data_dir ../data/canonical --indices_dir ../data/split-indices --dossier2title_dir ../data/helpers --text_embeddings_dir ../data/text-embeddings
```

You can then generate "chronological" datasets, where edits are ordered according to the date of the dossiers:

```
cd 1-datasets
./1-split-chronologically.sh ../data/canonical/war-of-words-2-ep8.txt ../data/canonical/war-of-words-2-ep8-chronological.txt
```

To generate text embeddings for these, you can then run

```
cd 0-text-embeddings
python generate_embeddings.py --leg 8 --data_dir ../data/canonical --dossier2title_dir ../data/helpers --text_embeddings_dir ../data/text-embeddings --chronological
```

## Step 1: Process the Datasets

Map the text embeddings to the canonical datasets:

```
cd 1-datasets
./2-map-text-embeddings.sh ../data/canonical ../data/text-embeddings
```

Finally, process the datasets to create training sets and test sets:

```
mkdir pkl
./3-generate-datasets.sh ../data/canonical pkl
```

## Step 2: Train the Models

To train the models, you define an "experiment" in a JSON file (see examples in the `train-def` folder).
You then train all the models, as defined in the JSON files, by running:

```
mkdir trained-models
python train.py --definition train-def/ep8.json --data_dir path/to/processed/datasets --hyperparams_dir hyperparams --models trained-models
```

Run all three definitions (`ep7.json`, `ep8.json`, and `ep8-chronological.json`) to train all the models in the paper.

## Step 3: Evaluate the Models

Similarly to the training, you define "experiments" for evaluation.
You then evaluate all experiments by running:

```
mkdir results
python eval.py --definition eval-def/ep8.json --data_dir ../1-datasets/pkl --models_dir ../2-training/trained-models --save_results results
```

Run all four definitions (`ep7.json`, `ep8.json`, `ep8-chronological.json`, and `ep8-conflict_size.json`) to evaluate all experiments in the paper.

## Step 4: Analyze the Results

You finally reproduce the analysis in the paper by running the scripts in the folder `4-analysis`:

```
python results.py --results ../3-evaluation/results --save-as figures/results.pdf
python improvement.py --results ../3-evaluation/results --save-as figures/improvement.pdf
python parameter-analysis.py --model ../2-trained-models/ep8-all_features-latent-text.fit
python error-analysis.py --save-as figures/error-analysis.pdf
```

The interpretation of the latent features is in `4-analysis/notebooks/latent-features.ipynb`.
The interpretation of the text features is in `4-analysis/notebooks/text-features.ipynb`.
Each of the scripts above also have a corresponding notebook, so that the outputs is easily accessed through notebook readers (such as on GitHub).


## Requirements

This project requires **Python 3.6**.

## Citation

To cite this work, use:

```
@inproceedings{kristof2021war,
  author = {Kristof, Victor and Suresh, Aswin and Grossglauser, Matthias and Thiran, Patrick},
  title = {War of Words II: Enriched Models for Law-Making Processes},
  year = {2021},
  booktitle = {Proceedings of The Web Conference 2021},
  TODO: pages = {2803–2809},
  numpages = {12},
  location = {Ljubljana, Solvenia},
  series = {WWW '21}
}
```
