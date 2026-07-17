import logging
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class SearchFilters:
    asset: Optional[str] = None
    category: Optional[str] = None
    publisher: Optional[str] = None
    source: Optional[str] = None
    topics: Optional[List[str]] = None


class VectorService:
    """
    Handles document embeddings, FAISS indexing,
    persistence and similarity search.
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SEARCH_MULTIPLIER = 10
    MIN_SEARCH_K = 50
    
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
        self.document_embeddings: Dict[str, np.ndarray] = {}

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
    
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a query string.
        """

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype(np.float32)
    
    def _build_document_text(self, document: Dict) -> str:
        """
        Convert a news document into a single string
        for embedding.
        """

        return f"""
            Asset:
            {document.get("asset", "")}

            Category:
            {document.get("category", "")}

            Title:
            {document.get("title", "")}

            Summary:
            {document.get("summary", "")}

            Publisher:
            {document.get("publisher", "")}

            Topics:
            {", ".join(document.get("topics", []))}

            Published:
            {document.get("published", "")}
        """
    
    def _create_metadata_index(self):

        return {
            "asset": defaultdict(set),
            "category": defaultdict(set),
            "publisher": defaultdict(set),
            "source": defaultdict(set),
            "topics": defaultdict(set),
        }
    
    def _index_metadata(self, document: Dict) -> None:
        """
        Index document metadata for fast filtering.
        """

        doc_id = document["id"]

        # Single-value fields
        for field in ("asset", "category", "publisher", "source"):

            value = document.get(field)

            if value:
                self.metadata_index[field][value].add(doc_id)

        # Multi-value field
        for topic in document.get("topics", []):

            self.metadata_index["topics"][topic].add(doc_id)

    def _get_candidate_documents(
        self,
        filters: Optional[SearchFilters] = None,
    ) -> Optional[Set[str]]:
        """
        Return candidate document ids.
        Pre filtering based on metadata can speed up the search.
        if filters is None, return None (no filtering).
        """

        if filters is None:
            return None
        
        candidate_sets = []

        if filters.asset:
            candidate_sets.append(
                self.metadata_index["asset"].get(
                    filters.asset,
                    set(),
                )
            )
        
        if filters.category:
            candidate_sets.append(
                self.metadata_index["category"].get(
                    filters.category,
                    set(),
                )
            )
        
        if filters.publisher:
            candidate_sets.append(
                self.metadata_index["publisher"].get(
                    filters.publisher,
                    set(),
                )
            )
        
        if filters.source:
            candidate_sets.append(
                self.metadata_index["source"].get(
                    filters.source,
                    set(),
                )
            )
        
        #Topics is a multi-value field, so we need to union the sets
        topic_candidates = set()

        for topic in filters.topics or []:
            topic_candidates |= self.metadata_index["topics"].get(
                topic,
                set(),
            )

        if topic_candidates:
            candidate_sets.append(topic_candidates)

        if not candidate_sets:
            return None
        if candidate_sets and any(len(s) == 0 for s in candidate_sets):
            return set()
        
        #Intersect all candidate sets
        candidates = candidate_sets[0].copy()

        for candidate_set in candidate_sets[1:]:
            candidates &= candidate_set

        return candidates
    
    def _should_use_temp_index(
        self,
        candidate_count: int,
    ) -> bool:
        """
        Decide whether to search a temporary index.
        """

        threshold = max(
            50,
            int(self.count() * 0.20),
        )

        return candidate_count <= threshold
    
    def _build_temp_index(
        self,
        candidate_ids: Set[str],
    ) -> tuple[faiss.IndexFlatIP, List[str]]:
        """
        Build a temporary FAISS index
        from the candidate documents.
        """

        temp_index = faiss.IndexFlatIP(
            self.dimension
        )

        temp_mapping = []
        vectors = []

        for doc_id in candidate_ids:

            embedding = self.document_embeddings.get(doc_id)

            if embedding is None:
                continue

            vectors.append(embedding)
            temp_mapping.append(doc_id)

        if vectors:

            temp_index.add(
                np.vstack(vectors)
            )

        return temp_index, temp_mapping
    
    def _search_temp_index(
        self,
        embedding: np.ndarray,
        temp_index: faiss.IndexFlatIP,
        k: int,
    ):
        """
        Execute a similarity search
        over a temporary index.
        """
        k = min(
            k,
            temp_index.ntotal,
        )

        return temp_index.search(
            embedding,
            k,
        )
    
    def _search_faiss(
        self,
        embedding: np.ndarray,
        index: faiss.IndexFlatIP,
        mapping: List[str],
        k: int,
        expand_search: bool = True,
    ):
        """
        Execute a FAISS similarity search.
        """

        if expand_search:
            search_k = max(
                k * self.SEARCH_MULTIPLIER,
                self.MIN_SEARCH_K,
            )

        else:
            search_k = k

        scores, indices = index.search(
            embedding,
            search_k,
        )

        return scores, indices, mapping
    
    def _build_results(
        self,
        scores,
        indices,
        mapping: List[str],
        min_score: float,
    ) -> List[Dict]:
        """
        Convert FAISS results into ranked documents.
        """

        results = []
        rank = 1

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            if score < min_score:
                continue

            doc_id = mapping[idx]

            document = self.documents[doc_id].copy()

            document["rank"] = rank
            document["score"] = round(float(score), 3)

            results.append(document)

            rank += 1

        return results
    
    def _filter_documents(
        self,
        documents: List[Dict],
        filters: SearchFilters,
    ) -> List[Dict]:
        """
        Apply metadata filters after semantic search.

        Used when the search is performed on the complete
        FAISS index instead of a temporary filtered index.
        """

        if filters is None:
            return documents

        filtered = []

        for document in documents:

            # Asset
            if filters.asset:
                if document.get("asset") != filters.asset:
                    continue

            # Category
            if filters.category:
                if document.get("category") != filters.category:
                    continue

            # Publisher
            if filters.publisher:
                if document.get("publisher") != filters.publisher:
                    continue

            # Source
            if filters.source:
                if document.get("source") != filters.source:
                    continue

            # Topics
            if filters.topics:

                document_topics = set(
                    document.get("topics", [])
                )

                if not document_topics.intersection(filters.topics):
                    continue

            filtered.append(document)

        #Rerank the filtered documents
        for rank, document in enumerate(filtered, start=1):
            document["rank"] = rank

        return filtered

    def add_documents(self, documents: List[Dict]) -> int:

        new_doc_ids = []
        texts = []

        for document in documents:

            document_id = document.get("id")

            if not document_id:
                continue

            if document_id in self.documents:
                continue
            
            # Store the document
            self.documents[document_id] = document
            # Update metadata
            self._index_metadata(document)
            # Build text for embedding
            texts.append(self._build_document_text(document))
            #keep track of new documents
            new_doc_ids.append(document_id)

        if not new_doc_ids:
            return 0

        # Generate embeddings and add to FAISS index
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        #Save embeddings for pre filtering and other uses
        for document_id, embedding in zip(new_doc_ids, embeddings):
            self.document_embeddings[document_id] = embedding.astype(np.float32)

        # Store embeddings in permanent FAISS index
        self.index.add(
            embeddings.astype(np.float32)
        )

        #Build FAISS mappings
        start_idx = len(self.faiss_to_doc)

        for offset, document_id in enumerate(new_doc_ids):

            faiss_idx = start_idx + offset

            self.faiss_to_doc.append(document_id)
            self.doc_to_faiss[document_id] = faiss_idx

        assert len(self.faiss_to_doc) == self.index.ntotal
        logger.debug(
            "Added %d documents to vector store.",
            len(new_doc_ids)
        )

        return len(new_doc_ids)
    
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
        k: int = 5,
        min_score: float = 0.40,
        filters: Optional[SearchFilters] = None,
    ):

        if self.index.ntotal == 0:
            return []

        embedding = self._embed_query(query)

        candidate_ids = self._get_candidate_documents(filters)

        # Metadata filters returned no possible documents
        if candidate_ids == set():
            return []

        # ---------------------------------------------------------
        # Strategy 1: No metadata filters
        # ---------------------------------------------------------
        if candidate_ids is None:

            scores, indices, mapping = self._search_faiss(
                embedding=embedding,
                index=self.index,
                mapping=self.faiss_to_doc,
                k=k,
                expand_search=False,
            )

            logger.debug(
                "Search strategy: permanent index"
            )

            # print(f"Search strategy: permanent index ({self.count()} candidates)")
       
            return self._build_results(
                scores=scores,
                indices=indices,
                mapping=mapping,
                min_score=min_score,
            )

        # ---------------------------------------------------------
        # Strategy 2: Small candidate set
        # Build a temporary FAISS index
        # ---------------------------------------------------------
        if self._should_use_temp_index(len(candidate_ids)):

            temp_index, temp_mapping = self._build_temp_index(
                candidate_ids,
            )

            scores, indices, mapping = self._search_faiss(
                embedding=embedding,
                index=temp_index,
                mapping=temp_mapping,
                k=k,
                expand_search=False,
            )

            logger.debug(
                "Search strategy: temporary index (%d candidates)",
                len(candidate_ids),
            )
            # print(f"Search strategy: temporary index ({len(candidate_ids)} candidates)")

            return self._build_results(
                scores=scores,
                indices=indices,
                mapping=mapping,
                min_score=min_score,
            )

        # ---------------------------------------------------------
        # Strategy 3: Large candidate set
        # Permanent FAISS + post-filter
        # ---------------------------------------------------------
        scores, indices, mapping = self._search_faiss(
            embedding=embedding,
            index=self.index,
            mapping=self.faiss_to_doc,
            k=k,
            expand_search=True,
        )

        results = self._build_results(
            scores=scores,
            indices=indices,
            mapping=mapping,
            min_score=min_score,
        )

        logger.debug(
            "Search strategy: permanent index + post-filter (%d candidates)",
            len(candidate_ids),
        )

        #print(f"Search strategy: permanent index + post-filter ({len(candidate_ids)} candidates)")
        
        return self._filter_documents(
            results,
            filters,
        )
    
    def retrieve_context(self, query: str, filters: Optional[SearchFilters] = None, k: int = 5) -> str:

        return None
    
    def save(self):

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        state = {
            "version": 2,
            "embedding_model": self.EMBEDDING_MODEL,
            "dimension": self.dimension,
            "documents": self.documents,
            "document_embeddings": self.document_embeddings,
            "faiss_to_doc": self.faiss_to_doc,
            "doc_to_faiss": self.doc_to_faiss,
            "metadata_index": {
                key: dict(value)
                for key, value in self.metadata_index.items()
            }
        }

        with open(self.storage_path_path, "wb") as f:
            pickle.dump(state, f)

    def load(self):

        if self.index_path.exists():
            self.index = faiss.read_index(
                str(self.index_path)
            )

        if not self.storage_path_path.exists():
            return

        with open(self.storage_path_path, "rb") as f:
            state = pickle.load(f)
        
        # Restore state
        self.documents = state.get(
            "documents",
            {}
        )

        self.document_embeddings = state.get(
            "document_embeddings",
            {}
        )

        self.faiss_to_doc = state.get(
            "faiss_to_doc",
            []
        )

        self.doc_to_faiss = state.get(
            "doc_to_faiss",
            {}
        )

        #Restore metadata index
        self.metadata_index = self._create_metadata_index()

        saved_metadata = state.get(
            "metadata_index",
            {}
        )

        for category, values in saved_metadata.items():

            for value, document_ids in values.items():

                self.metadata_index[category][value] = set(document_ids)

    def count(self) -> int:
        return self.index.ntotal

    def stats(self):
        return {
            "documents": self.count(),
            "embedding_model": self.EMBEDDING_MODEL,
            "dimension": self.dimension,
        }

    def clear(self):

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.documents = {}

        self.faiss_to_doc = []

        self.doc_to_faiss = {}

        self.metadata_index = self._create_metadata_index()

        self.save()

    