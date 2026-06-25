"""Task 1 — Ingest NBME rubric concepts into ChromaDB.

Reads NBME_PN_HISTORY_FEATURES.txt, embeds each concept with
sentence-transformers, and stores in a persistent ChromaDB collection.

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
    parts = feature_text.split("-OR-")
    return [p.replace("-", " ").strip() for p in parts]


def build_document(synonyms: list[str]) -> str:
    primary = synonyms[0]
    all_variants = ", ".join(synonyms)
    return f"Concept: {primary}\nSynonyms: {all_variants}"


def main() -> None:
    print("ClinNoteRAG — Concept Ingestion (new NBME dataset)")
    print("=" * 50)

    df = pd.read_csv(DATA_DIR / "NBME_PN_HISTORY_FEATURES.txt", sep="|")
    print(f"Loaded {len(df)} concepts from NBME_PN_HISTORY_FEATURES.txt\n")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection("nbme_concepts")
    except Exception:
        pass
    collection = client.create_collection(
        name="nbme_concepts",
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for case_num in sorted(df["CASE_NUM"].unique()):
        case_df = df[df["CASE_NUM"] == case_num].reset_index(drop=True)

        ids, documents, metadatas = [], [], []

        for _, row in case_df.iterrows():
            feature_num = str(int(row["FEATURE_NUM"]))
            synonyms = parse_synonyms(str(row["FEATURE_TEXT"]))
            doc = build_document(synonyms)

            ids.append(f"{case_num}_{feature_num}")
            documents.append(doc)
            metadatas.append({
                "case_num": int(case_num),
                "feature_num": feature_num,
                "feature_text": str(row["FEATURE_TEXT"]),
            })

        vectors = embed(documents)
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors,
        )

        count = len(ids)
        total += count
        print(f"  Case {case_num}: {count} concepts ingested")

    print(f"\nTotal: {total} concepts stored in ChromaDB at '{CHROMA_PATH}'")
    print("Done. Run 'python services/agent.py' to smoke-test the agent.")


if __name__ == "__main__":
    main()
