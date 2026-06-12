"""Task 1 — Ingest NBME rubric concepts into ChromaDB.

Reads features.csv, embeds each concept with sentence-transformers,
and stores in a persistent ChromaDB collection.

Run:
    python scripts/ingest_concepts.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

import pandas as pd
import chromadb
from django.conf import settings

from services.embedder import embed


DATA_DIR = settings.NBME_DATA_DIR
CHROMA_PATH = settings.CHROMA_DB_PATH


def parse_synonyms(feature_text: str) -> list[str]:
    """Split 'A-B-OR-C-D' into ['A B', 'C D'] and clean hyphens."""
    parts = feature_text.split("-OR-")
    return [p.replace("-", " ").strip() for p in parts]


def build_document(synonyms: list[str]) -> str:
    """Build the text stored in ChromaDB for one concept."""
    primary = synonyms[0]
    all_variants = ", ".join(synonyms)
    return f"Concept: {primary}\nSynonyms: {all_variants}"


def main() -> None:
    print("ClinNoteRAG — Concept Ingestion")
    print("=" * 50)

    df = pd.read_csv(DATA_DIR / "features.csv")
    print(f"Loaded {len(df)} concepts from features.csv\n")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Wipe and recreate to ensure clean state on re-run
    try:
        client.delete_collection("nbme_concepts")
    except Exception:
        pass
    collection = client.create_collection(
        name="nbme_concepts",
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for case_num in range(10):
        case_df = df[df["case_num"] == case_num].reset_index(drop=True)
        if case_df.empty:
            continue

        ids, documents, metadatas, embeddings = [], [], [], []

        for _, row in case_df.iterrows():
            feature_num = str(row["feature_num"])
            synonyms = parse_synonyms(str(row["feature_text"]))
            doc = build_document(synonyms)

            ids.append(f"{case_num}_{feature_num}")
            documents.append(doc)
            metadatas.append({
                "case_num": case_num,
                "feature_num": feature_num,
                "feature_text": str(row["feature_text"]),
            })

        vectors = embed(documents)
        embeddings = vectors

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        count = len(ids)
        total += count
        print(f"  Case {case_num}: {count} concepts ingested")

    print(f"\nTotal: {total} concepts stored in ChromaDB at '{CHROMA_PATH}'")
    print("Done. Run 'python services/agent.py' to smoke-test the agent.")


if __name__ == "__main__":
    main()
