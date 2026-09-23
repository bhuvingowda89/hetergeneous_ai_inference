"""Tokenizer-aware deterministic prompt construction for GPU runs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


class PromptConstructionError(RuntimeError):
    """Raised when an exact tokenizer-length prompt cannot be constructed."""


@dataclass(frozen=True)
class ExactPrompt:
    text: str
    requested_tokens: int
    actual_tokens: int


@lru_cache(maxsize=8)
def load_tokenizer(tokenizer_identifier: str, revision: str | None = None) -> Any:
    """Load a Hugging Face tokenizer lazily for the vLLM backend."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise PromptConstructionError(
            "exact tokenizer-level prompts require transformers in the GPU environment"
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_identifier, revision=revision)


def tokenizer_identifier(model: str, tokenizer: str | None) -> str:
    """Return the effective tokenizer identifier for a vLLM config."""
    return tokenizer or model


def encode_count(tokenizer: Any, text: str) -> int:
    """Return token count without adding tokenizer-level special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def construct_exact_prompt(tokenizer: Any, token_count: int) -> ExactPrompt:
    """Construct text whose tokenizer length is exactly ``token_count``.

    The construction is deterministic: it searches token IDs in ascending order,
    selects a token that round-trips as one non-special token, repeats that token
    ID, decodes the sequence, and verifies the final encoded length.
    """
    if token_count < 0:
        raise PromptConstructionError(f"requested token count must be non-negative: {token_count}")
    if token_count == 0:
        return ExactPrompt(text="", requested_tokens=0, actual_tokens=0)

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        vocab = tokenizer.get_vocab()
        candidate_ids = sorted(vocab.values())
    else:
        candidate_ids = range(vocab_size)

    candidate_pieces: list[str] = []
    for token_id in candidate_ids:
        if token_id in special_ids:
            continue
        try:
            piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except Exception:
            continue
        if not piece or not piece.strip() or "\x00" in piece:
            continue
        if encode_count(tokenizer, piece) != 1:
            continue
        candidate_pieces.append(piece)
        text = tokenizer.decode([token_id] * token_count, clean_up_tokenization_spaces=False)
        actual = encode_count(tokenizer, text)
        if actual == token_count:
            return ExactPrompt(text=text, requested_tokens=token_count, actual_tokens=actual)

    text = ""
    actual = 0
    while actual < token_count:
        for piece in candidate_pieces:
            candidate = text + piece
            candidate_count = encode_count(tokenizer, candidate)
            if candidate_count == actual + 1:
                text = candidate
                actual = candidate_count
                break
        else:
            break
    if actual == token_count:
        return ExactPrompt(text=text, requested_tokens=token_count, actual_tokens=actual)

    raise PromptConstructionError(
        f"could not construct an exact {token_count}-token prompt for tokenizer "
        f"{getattr(tokenizer, 'name_or_path', '<unknown>')}"
    )


def resolve_hf_snapshot_revision(model_or_tokenizer: str, revision: str | None) -> str | None:
    """Resolve a cached Hugging Face snapshot revision/commit hash when available."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=model_or_tokenizer,
                revision=revision,
                local_files_only=True,
            )
        )
    except Exception:
        return None
    if snapshot_path.parent.name == "snapshots":
        return snapshot_path.name
    return None
