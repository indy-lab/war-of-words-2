import argparse
import json


def load_embeddings(path):
    with open(path) as f:
        return [[float(e) for e in line.split()] for line in f.readlines()]


def main(args):
    with open(args.canonical) as f:
        conflicts = [json.loads(line) for line in f.readlines()]

    # Load edit embeddings.
    edit_embeddings = load_embeddings(args.edit_embedding)
    # Load title embeddings.
    title_embeddings = load_embeddings(args.title_embedding)

    c = 0
    for conflict in conflicts:
        for edit in conflict:
            c += 1

    i = 0
    with open(args.output, 'w') as f:
        for conflict in conflicts:
            # Open JSON list.
            f.write('[')
            for c, edit in enumerate(conflict):
                # Add embedding.
                edit['edit-embedding'] = edit_embeddings[i]
                edit['title-embedding'] = title_embeddings[i]
                # Write edit with embedding.
                f.write(json.dumps(edit))
                # Write separator if it is not the last edit.
                if c + 1 < len(conflict):
                    f.write(',')
                # Move to next embedding.
                i += 1
            # Close JSON list and move to next line.
            f.write(']')
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--canonical', help='Input path to canonical dataset')
    parser.add_argument(
        '--edit-embedding', help='Path to CSV of edit text embedding'
    )
    parser.add_argument(
        '--title-embedding', help='Path to CSV of title text embedding'
    )
    parser.add_argument(
        '--output', help='Output path to canonical dataset with embedding'
    )
    args = parser.parse_args()
    main(args)
