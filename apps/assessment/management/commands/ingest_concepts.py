"""Django management command: ingest NBME concepts into ChromaDB.

Usage:
    python manage.py ingest_concepts          # ingest (skip if collection exists)
    python manage.py ingest_concepts --force  # wipe and re-ingest
"""

import chromadb
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from services.embedder import embed


def parse_synonyms(feature_text: str) -> list[str]:
    parts = feature_text.split("-OR-")
    return [p.replace("-", " ").strip() for p in parts]


def build_document(synonyms: list[str]) -> str:
    primary = synonyms[0]
    all_variants = ", ".join(synonyms)
    return f"Concept: {primary}\nSynonyms: {all_variants}"


class Command(BaseCommand):
    help = "Ingest NBME rubric concepts into ChromaDB"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Delete and re-create the collection even if it already exists",
        )

    def handle(self, *args, **options):
        data_dir = settings.NBME_DATA_DIR
        chroma_path = settings.CHROMA_DB_PATH

        self.stdout.write("ClinNoteRAG — Concept Ingestion")
        self.stdout.write("=" * 50)

        df = pd.read_csv(data_dir / "NBME_PN_HISTORY_FEATURES.txt", sep="|")
        self.stdout.write(f"Loaded {len(df)} concepts from NBME_PN_HISTORY_FEATURES.txt\n")

        client = chromadb.PersistentClient(path=chroma_path)

        if options["force"]:
            try:
                client.delete_collection("nbme_concepts")
                self.stdout.write("Deleted existing collection.")
            except Exception:
                pass
        else:
            try:
                existing = client.get_collection("nbme_concepts")
                count = existing.count()
                self.stdout.write(
                    self.style.WARNING(
                        f"Collection already exists with {count} items. "
                        "Use --force to re-ingest."
                    )
                )
                return
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

            self.stdout.write(f"  Case {case_num}: {len(ids)} concepts ingested")
            total += len(ids)

        self.stdout.write(
            self.style.SUCCESS(f"\nDone. {total} concepts stored in ChromaDB at '{chroma_path}'")
        )
