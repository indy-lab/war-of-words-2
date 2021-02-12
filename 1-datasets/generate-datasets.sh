#! /bin/sh

# This scripts is a helper to preprocess datasets.
#
# It requires the following arguments:
# - $1: path to folder of "canonical" dataset
# - $2: output folder of pickle files
#
# We define the following tasks:
#  - Task 1: Predict new edits
#  - Task 2: Predict edits on new dossiers

# Check arguments.
if [ -z "$1" ]
  then
    echo "Missing path to folder of canonical dataset"
    exit 1
fi
if [ -z "$2" ]
  then
    echo "Missing path to output folder of pickle files"
    exit 1
fi

# TASK 1: EXPLICIT FEATURES

# Preprocessing parameters.
legs=(7 8)
exps=("no_features" "mep_features" "edit_features" "dossier_features" "rapporteur_advantage" "all_features")
thr=10
split=0.9
seed=0
for leg in "${legs[@]}"; do
    canonical=$1/war-of-words-2-ep$leg.txt
    for exp in "${exps[@]}"; do
        dataset=ep$leg-$exp
        echo "Generating $dataset..."
        if [ ! -f "$2/$dataset-train.pkl" ]; then
            output=$2/$dataset.pkl
            python $exp.py $canonical $output --threshold $thr --split $split --seed $seed
        else
            echo "Already generated."
        fi
    done
done

# TASK 1: TEXT FEATURES

# Preprocessing parameters.
legs=(7 8)
exps=("no_features" "all_features")
text="text"
thr=10
split=0.9
seed=0
for leg in "${legs[@]}"; do
    canonical=$1/war-of-words-2-ep$leg-with_text_embedding.txt
    for exp in "${exps[@]}"; do
        dataset=ep$leg-$exp-$text
        echo "Generating $dataset..."
        if [ ! -f "$2/$dataset-train.pkl" ]; then
            output=$2/$dataset.pkl
            python $exp.py $canonical $output --threshold $thr --split $split --seed $seed --text-features
        else
            echo "Already generated."
        fi
    done
done

# TASK 2: EXPLICIT FEATURES

# Preprocessing parameters.
leg=8
split=125557
exps=("no_features" "all_features")
order="chronological"
canonical=$1/war-of-words-2-ep$leg-$order.txt
# Generate the datasets.
for exp in "${exps[@]}"; do
    dataset=ep$leg-$exp-$order
    echo "Generating $dataset..."
    if [ ! -f "$2/$dataset-train.pkl" ]; then
        output=$2/$dataset.pkl
        python $exp.py $canonical $output --split $split
    else
        echo "Already generated."
    fi
done

# TASK 2: TEXT FEATURES

# Preprocessing parameters.
leg=8
split=125557
exps=("no_features" "all_features")
text="text"
order="chronological"
canonical=$1/war-of-words-2-ep$leg-with_text_embedding-$order.txt
# Generate the datasets.
for exp in "${exps[@]}"; do
    dataset=ep$leg-$exp-$text-$order
    echo "Generating $dataset..."
    if [ ! -f "$2/$dataset-train.pkl" ]; then
        output=$2/$dataset.pkl
        python $exp.py $canonical $output --split $split --text-features
    else
        echo "Already generated."
    fi
done
