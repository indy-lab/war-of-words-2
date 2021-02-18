# War of Words II: Enriched Models for Law-Making Processes

Data and code for

> Victor Kristof, Aswin Suresh, Matthias Grossglauser, Patrick Thiran, [_War of Words II: Enriched Models of Law-Making Processes_](), The Web Conference 2021, April 19-23, 2021, Ljubljana, Slovenia.

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

If you don't want to generate the text embeddings from scratch, put these in

```
mkdir data/text-embedding
```

## Step 0: Learn the Text Embeddings

Start by processing the canonical datasets

## Step 1: Process the Datasets

Start by generating the "chronological" datasets, where edits are ordered according to the date of the dossiers:

```
cd 1-datasets
./1-split-chronologically.sh ../data/canonical/war-of-words-ep8.txt ../data/canonical/war-of-words-ep8-chronological.txt
```

Then map the text embeddings to the canonical datasets:

```
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

```

```
