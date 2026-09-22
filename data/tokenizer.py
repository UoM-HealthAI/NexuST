import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import OrderedDict


class GeneVocab:
    """
    Gene vocabulary for encoding/decoding gene names to token IDs.
    Vocab file should already contain <pad>: 0.
    """
    def __init__(self, gene_dict: Dict[str, int], oov_token: str = "<pad>"):
        """
        Args:
            gene_dict: Dict of gene->id mappings (including <pad>)
            oov_token: Token to return for out-of-vocabulary genes (default: <pad>)
        """
        self.vocab = OrderedDict()
        self.itos = {}  # index to string

        for gene, idx in sorted(gene_dict.items(), key=lambda x: x[1]):
            self.vocab[gene] = idx
            self.itos[idx] = gene

        # OOV handling
        self.oov_token = oov_token
        self.oov_index = self.vocab.get(oov_token, 0)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]):
        """Load vocabulary from JSON file"""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("r") as f:
            gene_dict = json.load(f)
        return cls(gene_dict)

    def encode(self, gene: str) -> int:
        """Encode gene name to token ID. Returns oov_index for unknown genes."""
        return self.vocab.get(gene, self.oov_index)

    def decode(self, idx: int) -> str:
        """Decode token ID to gene name. Returns oov_token for unknown IDs."""
        return self.itos.get(idx, self.oov_token)

    def __contains__(self, key: str) -> bool:
        """Check if gene is in vocabulary"""
        return key in self.vocab

    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab.get("<pad>", 0)


class MetadataVocab:
    """
    Vocabulary for metadata fields (organ, platform, etc.)
    Handles nested structure: {"organ": {"unknown": 0, "brain": 1, ...}, "platform": {...}}
    Each field should have "unknown": 0 as OOV token.
    """
    def __init__(self, vocab_dict: Dict[str, Dict[str, int]] = None, oov_token: str = "unknown"):
        """
        Args:
            vocab_dict: Nested dict of field -> {value: id} mappings
            oov_token: Token to return for out-of-vocabulary values (default: "unknown")
        """
        self.vocab = vocab_dict or {}
        self.itos = {
            field: {v: k for k, v in mapping.items()}
            for field, mapping in self.vocab.items()
        }

        # OOV handling (consistent with GeneVocab)
        self.oov_token = oov_token
        self.oov_index = 0  # "unknown" is always at index 0

    @classmethod
    def from_file(cls, file_path: Union[Path, str]):
        """Load vocabulary from JSON file"""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("r") as f:
            return cls(json.load(f))

    def encode(self, field: str, value: str) -> int:
        """Encode a metadata value to its ID. Returns oov_index (0) for unknown values."""
        if field not in self.vocab:
            return self.oov_index
        return self.vocab[field].get(value, self.oov_index)

    def decode(self, field: str, idx: int) -> str:
        """Decode an ID back to its string value. Returns oov_token for unknown IDs."""
        if field not in self.itos:
            return self.oov_token
        return self.itos[field].get(idx, self.oov_token)

    def get_vocab_size(self, field: str) -> int:
        """Get vocabulary size for a specific field"""
        return len(self.vocab.get(field, {}))

    def get_fields(self) -> List[str]:
        """Get list of all metadata fields"""
        return list(self.vocab.keys())

    def __contains__(self, field: str) -> bool:
        return field in self.vocab


class Tokenizer:
    """
    Tokenizer for NexuST spatial transcriptomics data.
    Provides unified interface for gene and metadata encoding/decoding.
    """
    def __init__(self, vocab_file: Optional[str] = None, metadata_vocab_file: Optional[str] = None):
        """
        Args:
            vocab_file: Path to vocabulary JSON file (uses default if None)
        """
        gene_vocab_file = Path(vocab_file) if vocab_file is not None else Path(__file__).with_name("gene_vocab.json")
        metadata_vocab_file = Path(metadata_vocab_file) if metadata_vocab_file is not None else Path(__file__).with_name("metadata_vocab.json")

        self.gene_vocab = GeneVocab.from_file(gene_vocab_file)
        self.metadata_vocab = MetadataVocab.from_file(metadata_vocab_file)

        self.gene_vocab_size = len(self.gene_vocab)
        self.organ_vocab_size = self.metadata_vocab.get_vocab_size("organ")
        self.platform_vocab_size = self.metadata_vocab.get_vocab_size("platform")
        self.pad_token_id = self.gene_vocab.pad_token_id

        print(f"Loaded gene vocabulary with {self.gene_vocab_size} tokens (pad_token_id={self.pad_token_id})")
        print(f"Loaded metadata vocabulary with fields: {self.metadata_vocab.get_fields()}")

    def encode_gene(self, gene: str) -> int:
        """Encode gene name to token ID"""
        return self.gene_vocab.encode(gene)

    def decode_gene(self, idx: int) -> str:
        """Decode token ID to gene name"""
        return self.gene_vocab.decode(idx)

    def encode_metadata(self, field: str, value: str) -> int:
        """Encode a metadata value to its ID"""
        return self.metadata_vocab.encode(field, value)

    def decode_metadata(self, field: str, idx: int) -> str:
        """Decode a metadata ID back to string"""
        return self.metadata_vocab.decode(field, idx)


def get_tokenizer(vocab_file=None, metadata_vocab_file=None) -> Tokenizer:
    return Tokenizer(vocab_file, metadata_vocab_file)


if __name__ == "__main__":
    tokenizer = get_tokenizer()

    # Test gene encoding
    print("\n--- Gene Encoding Test ---")
    test_genes = ["GAPDH", "ACTB", "TP53", "UNKNOWN_GENE"]
    for gene in test_genes:
        if gene in tokenizer.gene_vocab:
            gene_id = tokenizer.encode_gene(gene)
            decoded = tokenizer.decode_gene(gene_id)
            print(f"{gene} -> {gene_id} -> {decoded}")
        else:
            gene_id = tokenizer.encode_gene(gene)
            print(f"{gene} not in vocabulary -> {gene_id} (oov)")

    # Test metadata encoding
    print("\n--- Metadata Encoding Test ---")
    test_metadata = [
        ("organ", "brain"),
        ("organ", "liver"),
        ("organ", "unknown_organ"),
        ("platform", "merfish"),
        ("platform", "xenium"),
    ]
    for field, value in test_metadata:
        idx = tokenizer.encode_metadata(field, value)
        decoded = tokenizer.decode_metadata(field, idx)
        print(f"{field}:{value} -> {idx} -> {decoded}")

    # Print vocab sizes
    print("\n--- Vocabulary Sizes ---")
    print(f"Gene vocab size: {tokenizer.gene_vocab_size}")
    print(f"Organ vocab size: {tokenizer.organ_vocab_size}")
