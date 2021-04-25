#!/bin/sh

# This scripts is a helper to preprocess datasets.
#
# It requires the following arguments:
# - $1: path to folder of "canonical" dataset
# - $2: path to folder of text embeddings
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
    echo "Missing path to folder of text embeddings"
    exit 1
fi
canonicaldir=$1
embeddingdir=$2

# TASK 1

legs=(7 8)
for leg in "${legs[@]}"; do
    canonical=$canonicaldir/war-of-words-2-ep$leg.txt
    editembedding="edit_embedding"
    titleembedding="title_embedding"
    dataset=war-of-words-2-ep$leg-with_text_embedding
    echo "Generating $dataset..."
    output=$canonicaldir/$dataset.txt
    if [ ! -f "$output" ]; then
        python map-text-embeddings.py \
            --canonical $canonical \
            --edit-embedding $embeddingdir/ep$leg-$editembedding.txt \
            --title-embedding $embeddingdir/ep$leg-$titleembedding.txt \
            --output $output
        echo "Saved to $output"
    else
        echo "Already generated."
    fi
done

# Map embeddings trained on the whole dataset.
dataset=war-of-words-2-ep$leg-with_text_embedding_for_parameter_analysis
echo "Generating $dataset..."
output=$canonicaldir/$dataset.txt
if [ ! -f "$output" ]; then
    python map-text-embeddings.py \
        --canonical $canonical \
        --edit-embedding $embeddingdir/parameter-analysis/ep$leg-$editembedding.txt \
        --title-embedding $embeddingdir/parameter-analysis/ep$leg-$titleembedding.txt \
        --output $output
    echo "Saved to $output"
else
    echo "Already generated."
fi


# TASK 2

leg=8
canonical=$canonicaldir/war-of-words-2-ep$leg-chronological.txt
editembedding="edit_embedding-chronological"
titleembedding="title_embedding-chronological"
dataset=war-of-words-2-ep$leg-with_text_embedding-chronological
echo "Generating $dataset..."
output=$canonicaldir/$dataset.txt
if [ ! -f "$output" ]; then
    python map-text-embeddings.py \
        --canonical $canonical \
        --edit-embedding $embeddingdir/ep$leg-$editembedding.txt \
        --title-embedding $embeddingdir/ep$leg-$titleembedding.txt \
        --output $output
    echo "Saved to $output"
else
    echo "Already generated."
fi
