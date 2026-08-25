"""Compare the local E5 backend with the official Transformers pooling path."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from app.retrieval.embeddings import SentenceTransformerEmbeddingBackend


MODEL_PATH = Path(r"D:\proje_Staj\models\embedding\multilingual-e5-base")


def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply the attention-mask-aware average pooling used by E5."""

    masked = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def reference_embeddings(tokenizer: AutoTokenizer, model: AutoModel,
                         texts: list[str]) -> np.ndarray:
    """Produce normalized embeddings through the official Transformers path."""

    encoded = tokenizer(
        texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        output = model(**encoded)
    pooled = average_pool(output.last_hidden_state, encoded["attention_mask"])
    return functional.normalize(pooled, p=2, dim=1).cpu().numpy()


def main() -> None:
    """Run Turkish and English retrieval comparisons without network access."""

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_PATH), local_files_only=True, trust_remote_code=False
    )
    model.eval()
    backend = SentenceTransformerEmbeddingBackend(MODEL_PATH, device="cpu")

    first = [
        "The inertial navigation system estimates position, velocity and orientation using onboard inertial sensors.",
        "The radar antenna operates in the microwave frequency range and transmits electromagnetic signals.",
        "The propulsion system generates thrust required for vehicle acceleration during flight.",
    ]
    second = [
        "The guidance system calculates steering commands to direct the vehicle toward the target.",
        "The structural frame carries mechanical loads during operation.",
        "The power supply distributes electrical energy to onboard subsystems.",
    ]
    cases = [
        ("tr_inertial", "ataletsel seyrüsefer sistemi", first),
        ("tr_guidance", "güdüm sistemi", second),
        ("en_inertial", "inertial navigation system", first),
        ("en_guidance", "guidance system", second),
    ]
    results: list[dict[str, object]] = []
    for name, query, passages in cases:
        prefixed = [f"query: {query}", *[f"passage: {text}" for text in passages]]
        reference = reference_embeddings(tokenizer, model, prefixed)
        backend_vectors = np.vstack(
            [backend.embed_query(query), backend.embed_documents(passages)]
        )
        reference_scores = reference[1:] @ reference[0]
        backend_scores = backend_vectors[1:] @ backend_vectors[0]
        results.append(
            {
                "name": name,
                "query": query,
                "reference_dimension": int(reference.shape[1]),
                "backend_dimension": int(backend_vectors.shape[1]),
                "reference_norms": np.linalg.norm(reference, axis=1).tolist(),
                "backend_norms": np.linalg.norm(backend_vectors, axis=1).tolist(),
                "reference_scores": reference_scores.tolist(),
                "backend_scores": backend_scores.tolist(),
                "reference_ranking": np.argsort(-reference_scores).tolist(),
                "backend_ranking": np.argsort(-backend_scores).tolist(),
                "maximum_absolute_embedding_difference": float(
                    np.max(np.abs(reference - backend_vectors))
                ),
            }
        )
    print(json.dumps({"model_path": str(MODEL_PATH), "cases": results},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
