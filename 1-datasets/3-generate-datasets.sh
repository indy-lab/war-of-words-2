#! /bin/sh

# This scripts is a helper to preprocess datasets.
#
# It requires the following arguments:
# - $1: path to folder of "canonical" dataset
# - $2: output folder of pickle files
# - $3: (option) path to folder of split indices
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
seed=0
for leg in "${legs[@]}"; do
    canonical=$1/war-of-words-2-ep$leg.txt
    train_indices=$3/ep$leg-train-indices.txt
    test_indices=$3/ep$leg-test-indices.txt
    for exp in "${exps[@]}"; do
        dataset=ep$leg-$exp
        echo "Generating $dataset..."
        if [ ! -f "$2/$dataset-train.pkl" ]; then
            output=$2/$dataset.pkl
            python $exp.py \
                $canonical $output \
                --threshold $thr \
                --seed $seed \
                --train-indices $train_indices \
                --test-indices $test_indices
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
seed=0
for leg in "${legs[@]}"; do
    canonical=$1/war-of-words-2-ep$leg-with_text_embedding.txt
    train_indices=$3/ep$leg-train-indices.txt
    test_indices=$3/ep$leg-test-indices.txt
    for exp in "${exps[@]}"; do
        dataset=ep$leg-$exp-$text
        echo "Generating $dataset..."
        if [ ! -f "$2/$dataset-train.pkl" ]; then
            output=$2/$dataset.pkl
            python $exp.py \
                $canonical \
                $output \
                --threshold $thr \
                --seed $seed \
                --text-features \
                --train-indices $train_indices \
                --test-indices $test_indices
        else
            echo "Already generated."
        fi
    done
done

# TASK 1: WHOLE DATASET FOR PARAMETER ANALYSIS

dataset=ep$leg-$exp-$text
echo "Generating $dataset on whole dataset..."
if [ ! -f "$2/$dataset-fit.pkl" ]; then
    output=$2/$dataset.pkl
    python $exp.py $canonical $output --threshold $thr
else
    echo "Already generated."
fi

# TASK 1: BASELINES

legs=(7 8)
kinds=("train" "test")
baselines=("naive" "random")
for leg in "${legs[@]}"; do
    for kind in "${kinds[@]}"; do
        for baseline in "${baselines[@]}"; do
            dataset=ep$leg-$baseline-$kind
            echo "Generating $dataset..."
            if [ ! -f "$2/$dataset.pkl" ]; then
                cp $2/ep$leg-no_features-$kind.pkl $2/$dataset.pkl
            else
                echo "Already generated."
            fi
        done
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

# TASK 2: BASELINES

leg=8
kinds=("train" "test")
baselines=("naive" "random")
for kind in "${kinds[@]}"; do
    for baseline in "${baselines[@]}"; do
        dataset=ep$leg-$baseline-$order-$kind
        echo "Generating $dataset..."
        if [ ! -f "$2/$dataset.pkl" ]; then
            cp $2/ep$leg-no_features-$order-$kind.pkl $2/$dataset.pkl
        else
            echo "Already generated."
        fi
    done
done

