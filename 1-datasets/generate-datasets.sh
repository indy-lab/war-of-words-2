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
exps=("no_features" "mep_features" "edit_features" "dossier_features" "all_features")
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

# # Preprocessing parameters.
# legs=(7 8)
# exps=("no_features" "all_features")
# text="fasttext_10_bigrams_corrected"
# thr=10
# split=0.9
# seed=0
# for leg in "${legs[@]}"; do
#     canonical=../../data/data/canonical/ep$leg-conflicts-notext-$text.json
#     for exp in "${exps[@]}"; do
#         output=pkl/ep$leg-$exp-text_corrected.pkl
#         python $exp.py $canonical $output --threshold $thr --split $split --seed $seed --text-features
#     done
# done

# TASK 2: EXPLICIT FEATURES

# # Preprocessing parameters.
# leg=8
# split=126435
# exps=("no_features" "all_features")
# order="random"
# canonical=../../data/data/canonical/ep$leg-conflicts-notext-by_dossier_$order.json
# # Generate the datasets.
# for exp in "${exps[@]}"; do
#     output=pkl/ep8-$exp-by_dossier_$order.pkl
#     python $exp.py $canonical $output --split $split
# done

# TASK 2: TEXT FEATURES

# # Preprocessing parameters.
# leg=8
# split=125557
# exps=("no_features" "all_features")
# text="fasttext_10_bigrams_corrected"
# canonical=../../data/data/canonical/ep$leg-conflicts-notext-$text-by_dossier.json
# # Generate the datasets.
# for exp in "${exps[@]}"; do
#     output=pkl/ep$leg-$exp-text_corrected-by_dossier.pkl
#     python $exp.py $canonical $output --split $split --text-features
# done

# # Adding text features for MEPs Only.
# exp="meps_only"
# text="tfidf"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="tfidf_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_5"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_5_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_10"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_10_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_25"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_25_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_50"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_50_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_10_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_100"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="fasttext_100_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="law2vec"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="meps_only"
# text="googlenews"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# Adding text features for All Features.

# exp="all_features"
# text="tfidf"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="tfidf_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_5"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_5_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_10"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_10_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_25"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_25_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_50"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_50_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_10_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_100"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# exp="all_features"
# text="fasttext_100_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed

# # Preprocessing parameters.
# leg=7
# thr=10
# split=0.9
# seed=0
# # Define datasets.
# exps=("no_features")
# text="fasttext_10_bigrams"
# # Generate the datasets with text embeddings.
# for exp in "${exps[@]}"; do
#     python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed
# done

# exp="all_features"
# text="fasttext_10_bigrams"
# python $exp.py ../../data/data/canonical/ep$leg-conflicts-notext-$text.json pkl/ep$leg-$exp-$text.pkl --text-features --threshold $thr --split $split --seed $seed
