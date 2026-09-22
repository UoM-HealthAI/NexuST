"""
Build gene vocabulary from corpus.

Usage:
python build_vocab.py \
    --corpus_dir datasets/pretrain \
    --output new_gene_vocab.json
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import scanpy as sc
from typing import List

SPECIAL_TOKENS = ["<pad>"]

def build_vocab(
    corpus_dir: str,
    output_path: str,
    special_tokens: List[str] = SPECIAL_TOKENS,
) -> dict:
    """
    Build gene vocabulary from all h5ad files in corpus.
    Only includes gene names, no special tokens (handled by tokenizer).

    Args:
        corpus_dir: Root directory containing tech subdirectories
        output_path: Path to save vocab JSON file

    Returns:
        Dict mapping gene_name to index
    """
    corpus_dir = Path(corpus_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all genes
    all_genes = set()
    tech_genes = {}

    techs = sorted([d.name for d in corpus_dir.iterdir() if d.is_dir()])
    print(f"Found technologies: {techs}")

    for tech in techs:
        tech_dir = corpus_dir / tech
        files = list(tech_dir.glob("**/*.h5ad"))
        tech_genes[tech] = set()

        for f in tqdm(files, desc=f"Scanning {tech}"):
            adata = sc.read_h5ad(f, backed='r')
            tech_genes[tech].update(adata.var_names)

        all_genes.update(tech_genes[tech])
        print(f"  {tech}: {len(tech_genes[tech])} genes from {len(files)} files")

    print(f"\nTotal unique genes: {len(all_genes)}")

    # Build vocab: special tokens first, then sorted genes
    vocab = {token: i for i, token in enumerate(special_tokens)}
    offset = len(special_tokens)
    vocab.update({gene: i + offset for i, gene in enumerate(sorted(all_genes))})

    # Save
    with open(output_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nVocab saved to {output_path}")
    print(f"  - Total vocab size: {len(vocab)}")

    # Print per-tech stats
    print("\nPer-technology gene counts:")
    for tech, genes in tech_genes.items():
        print(f"  - {tech}: {len(genes)}")

    # Print overlap stats
    if len(tech_genes) > 1:
        tech_list = list(tech_genes.keys())
        common = tech_genes[tech_list[0]]
        for tech in tech_list[1:]:
            common = common & tech_genes[tech]
        print(f"\nCommon genes across all techs: {len(common)}")

    return vocab


def main():
    parser = argparse.ArgumentParser(
        description='Build gene vocabulary from corpus',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--corpus_dir', '-c', type=str, required=True,
                       help='Root directory containing tech subdirectories')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output path for vocab JSON file')

    args = parser.parse_args()

    build_vocab(
        corpus_dir=args.corpus_dir,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
