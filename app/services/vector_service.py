import logging
from pathlib import Path
import pickle
from typing import List, Dict
from collections import defaultdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorService:
    """
    Handles document embeddings, FAISS indexing,
    persistence and similarity search.
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self):

        # -----------------------------
        # Paths
        # -----------------------------
        # project root
        BASE_DIR = Path(__file__).resolve().parents[2]
        self.storage_path = BASE_DIR / "data" / "vector_store"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_path / "news.index"
        self.storage_path_path = self.storage_path / "vector_store.pkl"

        # -----------------------------
        # Embedding model
        # -----------------------------
        self.model = SentenceTransformer(self.EMBEDDING_MODEL)

        self.dimension = self.model.get_sentence_embedding_dimension()

        # -----------------------------
        # FAISS index
        # -----------------------------
        self.index = faiss.IndexFlatIP(self.dimension)

        # -----------------------------
        # Document store
        # -----------------------------
        self.documents: Dict[str, Dict] = {}

        # -----------------------------
        # FAISS mappings
        # -----------------------------

        #Faiss position -> document id
        self.faiss_to_doc: List[str] = []

        #document id -> Faiss position
        self.doc_to_faiss: Dict[str, int] = {}

        # -----------------------------
        # Metadata indexes
        # -----------------------------
        self.metadata_index = self._create_metadata_index()
        
        self.load()
        logger.info("Loaded %d documents", self.count())
    
    def _build_document_text(self, document: Dict) -> str:
        """
        Convert a news document into a single string
        for embedding.
        """

        return f"""
            Ticker:
            {document.get("ticker","")}

            Title:
            {document.get("title","")}

            Summary:
            {document.get("summary","")}

            Publisher:
            {document.get("publisher","")}

            Published:
            {document.get("published","")}
            """
    
    def _create_metadata_index(self):

        return {
            "asset": defaultdict(set),
            "category": defaultdict(set),
            "publisher": defaultdict(set),
            "source": defaultdict(set),
            "topics": defaultdict(set),
        }
    
    def add_documents(self, documents: List[Dict]) -> int:

        new_documents = []

        for document in documents:

            document_id = document.get("id")

            if not document_id:
                continue

            if document_id in self.document_ids:
                continue

            self.document_ids.add(document_id)
            new_documents.append(document)

        if not new_documents:
            return 0

        texts = [
            self._build_document_text(doc)
            for doc in new_documents
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.index.add(
            embeddings.astype(np.float32)
        )

        self.metadata.extend(new_documents)

        return len(new_documents)
    
    def ingest_documents(self, documents):

        inserted = self.add_documents(documents)

        if inserted:
            self.save()
        
        logger.info(
            "Indexed %d new documents (%d total)",
            inserted,
            self.count()
        )

        return inserted
    
    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:

        if self.index.ntotal == 0:
            return []

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            embedding.astype(np.float32),
            k
        )

        MIN_SCORE = 0.40
        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1 or score < MIN_SCORE:
                continue

            document = self.metadata[idx].copy()

            document["score"] = round(float(score), 3)

            results.append(document)

        return results
    
    def save(self):

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "wb"
        ) as f:

            pickle.dump(
                self.metadata,
                f
            )

    def load(self):

        if self.index_path.exists():

            self.index = faiss.read_index(
                str(self.index_path)
            )

        if self.metadata_path.exists():

            with open(
                self.metadata_path,
                "rb"
            ) as f:

                self.metadata = pickle.load(f)
        
        if self.metadata_path.exists():

            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

            self.document_ids = {
                doc["id"]
                for doc in self.metadata
                if "id" in doc
            }

    def count(self) -> int:
        return self.index.ntotal

    def stats(self):
        return {
            "documents": self.count(),
            "embedding_model": self.EMBEDDING_MODEL,
            "dimension": self.dimension,
        }

    def clear(self):

        self.index = faiss.IndexFlatIP(self.dimension)

        self.metadata = []

        self.document_ids = set()

        self.save()

    