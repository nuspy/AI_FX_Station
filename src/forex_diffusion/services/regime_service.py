"""
RegimeService: cluster latents into regimes and build ANN index for fast nearest-neighbor queries.

- Uses sklearn.KMeans for clustering and hnswlib for ANN index.
- Persists regime labels into latents table via DBService.update_latent_labels_bulk.
- Saves ANN index to artifacts (regime_index.bin) and provides query_regime() to return label and neighbor info.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
import hnswlib
from loguru import logger

from .db_service import DBService


ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./artifacts/models")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "regime_index.bin")
MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "regime_mapping.json")


class RegimeService:
    def __init__(self, engine=None):
        self.db = DBService(engine=engine)
        self.engine = self.db.engine
        self.index: Optional[hnswlib.Index] = None
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}
        self.dim = None
        self.kmeans: Optional[KMeans] = None

    def _load_latents_all(self, limit: Optional[int] = None) -> Tuple[List[int], List[np.ndarray]]:
        """
        Load latents from DB (ids, vectors), returns lists.
        """
        meta = self.db.latents_tbl.metadata
        tbl = self.db.latents_tbl
        with self.engine.connect() as conn:
            stmt = tbl.select().order_by(tbl.c.ts_utc.desc())
            if limit:
                stmt = stmt.limit(limit)
            rows = conn.execute(stmt).fetchall()
            ids = []
            vecs = []
            for r in rows:
                # Support both SQLAlchemy Row with _mapping and plain tuple fallback.
                try:
                    mapping = getattr(r, "_mapping", None)
                    if mapping is not None:
                        row_id = mapping.get("id") if "id" in mapping else mapping.get("ID") if "ID" in mapping else mapping.get(0)
                        latent_json = mapping.get("latent_json") if "latent_json" in mapping else mapping.get("latent") if "latent" in mapping else mapping.get("latent_json")
                    else:
                        # tuple fallback: table layout is (id, symbol, timeframe, ts_utc, model_version, latent_json, ts_created_ms)
                        row_id = r[0] if len(r) > 0 else None
                        latent_json = r[5] if len(r) > 5 else None

                    if row_id is None:
                        # skip malformed row
                        continue
                    ids.append(int(row_id))
                    try:
                        vec = np.asarray(json.loads(latent_json), dtype=float) if latent_json else np.array([])
                    except Exception:
                        vec = np.array([])
                    vecs.append(vec)
                except Exception:
                    # best-effort: skip rows that cannot be parsed
                    continue
            # reverse to ascending time for stability
            ids = ids[::-1]
            vecs = vecs[::-1]
            return ids, vecs

    def fit_clusters_and_index(self, n_clusters: int = 8, index_space: str = "l2", ef_construction: int = 200, M: int = 16, limit: Optional[int] = None):
        """
        Fit KMeans on latents, assign labels and build HNSW index.
        """
        ids, vecs = self._load_latents_all(limit=limit)
        if not vecs:
            raise RuntimeError("No latents available to fit clusters")
        # filter non-empty vectors and compute dim
        filtered = [(i, v) for i, v in zip(ids, vecs) if v is not None and len(v) > 0]
        if not filtered:
            raise RuntimeError("No valid latent vectors found")
        ids_f, vecs_f = zip(*filtered)
        X = np.vstack(vecs_f).astype(np.float32)
        self.dim = X.shape[1]
        # clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
        labels = self.kmeans.labels_.tolist()
        # update DB labels
        mapping_rows = [{"id": int(_id), "regime_label": f"r{lab}"} for _id, lab in zip(ids_f, labels)]
        self.db.update_latent_labels_bulk(mapping_rows)
        # build HNSW index
        p = hnswlib.Index(space=index_space, dim=self.dim)
        num_elements = X.shape[0]
        p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
        p.add_items(X, np.arange(num_elements))
        p.set_ef(50)
        # save index and mapping (id->idx)
        p.save_index(INDEX_PATH)
        self.index = p
        self.id_to_idx = {int(_id): int(i) for i, _id in enumerate(ids_f)}
        self.idx_to_id = {int(i): int(_id) for i, _id in enumerate(ids_f)}
        # store last_indexed_id so incremental updates can append new latents
        last_indexed_id = int(ids_f[-1]) if len(ids_f) > 0 else None
        with open(MAPPING_PATH, "w", encoding="utf-8") as fh:
            json.dump({"id_to_idx": self.id_to_idx, "idx_to_id": self.idx_to_id, "last_indexed_id": last_indexed_id}, fh)

    def incremental_update(self, batch_size: int = 1000):
        """
        Incrementally add new latents (id > last_indexed_id) to existing HNSW index.
        If index not present, raises.
        """
        if not os.path.exists(MAPPING_PATH) or not os.path.exists(INDEX_PATH):
            raise RuntimeError("Index/mapping not found; run full build first")
        # load mapping to get last_indexed_id
        with open(MAPPING_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        last_id = meta.get("last_indexed_id", None)
        # load new latents with id > last_id
        meta_db = self.db.latents_tbl.metadata
        tbl = self.db.latents_tbl
        with self.engine.connect() as conn:
            if last_id is None:
                stmt = tbl.select().order_by(tbl.c.ts_utc.asc()).limit(batch_size)
            else:
                stmt = tbl.select().where(tbl.c.id > int(last_id)).order_by(tbl.c.ts_utc.asc()).limit(batch_size)
            rows = conn.execute(stmt).fetchall()
            if not rows:
                return {"updated": 0, "message": "no_new_latents"}
            new_ids = []
            new_vecs = []
            for r in rows:
                try:
                    vec = np.asarray(json.loads(r["latent_json"]), dtype=np.float32)
                except Exception:
                    continue
                if vec.size == 0:
                    continue
                new_ids.append(int(r["id"]))
                new_vecs.append(vec)
        if len(new_vecs) == 0:
            return {"updated": 0, "message": "no_valid_latents"}

        # load index and mapping
        self.load_index()
        # current count
        try:
            cur_count = int(self.index.get_current_count())
        except Exception:
            cur_count = len(self.idx_to_id)
        # add items
        data = np.vstack(new_vecs).astype(np.float32)
        start_idx = cur_count
        try:
            self.index.add_items(data, np.arange(start_idx, start_idx + data.shape[0]))
        except Exception as e:
            logger.exception("RegimeService.incremental_update: failed to add_items to index: {}", e)
            raise
        # update mappings
        for i, _id in enumerate(new_ids):
            idx = start_idx + i
            self.id_to_idx[int(_id)] = int(idx)
            self.idx_to_id[int(idx)] = int(_id)
        # persist index and mapping (update last_indexed_id)
        last_indexed_id = int(new_ids[-1])
        with open(MAPPING_PATH, "w", encoding="utf-8") as fh:
            json.dump({"id_to_idx": self.id_to_idx, "idx_to_id": self.idx_to_id, "last_indexed_id": last_indexed_id}, fh)
        try:
            self.index.save_index(INDEX_PATH)
        except Exception as e:
            logger.exception("RegimeService.incremental_update: failed to save index: {}", e)
        return {"updated": len(new_ids), "last_indexed_id": last_indexed_id}

    def load_index(self):
        """
        Load the prebuilt HNSW index and mapping if present.
        """
        if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
            raise RuntimeError("Index or mapping file not found")
        with open(MAPPING_PATH, "r", encoding="utf-8") as fh:
            m = json.load(fh)
            self.id_to_idx = {int(k): int(v) for k, v in m.get("id_to_idx", {}).items()}
            self.idx_to_id = {int(k): int(v) for k, v in m.get("idx_to_id", {}).items()}
        # infer dim from first latent
        # use DB to fetch one vector
        ids, vecs = self._load_latents_all(limit=1)
        if not vecs or len(vecs[0]) == 0:
            raise RuntimeError("Cannot infer dim for index")
        self.dim = len(vecs[0])
        p = hnswlib.Index(space="l2", dim=self.dim)
        p.load_index(INDEX_PATH)
        self.index = p

    def query_regime(self, query_vec: List[float], k: int = 10) -> Dict[str, Any]:
        """
        Query the ANN index and return regime label (majority) and neighbors.
        """
        if self.index is None:
            try:
                self.load_index()
            except Exception:
                return {"found": False, "reason": "index_not_loaded"}
        q = np.asarray(query_vec, dtype=np.float32)
        labels, distances = self.index.knn_query(q, k=k)
        ids = [int(self.idx_to_id.get(int(idx), -1)) for idx in labels[0]]
        # fetch labels from DB for these ids
        meta = self.db.latents_tbl.metadata
        tbl = self.db.latents_tbl
        with self.engine.connect() as conn:
            stmt = select(tbl.c.id, tbl.c.regime_label).where(tbl.c.id.in_(ids))
            rows = conn.execute(stmt).fetchall()
            label_map = {int(r[0]): (r[1] or "") for r in rows}
        neighbor_labels = [label_map.get(i, "") for i in ids]
        # majority vote ignoring empty
        votes = [l for l in neighbor_labels if l]
        if votes:
            from collections import Counter
            cnt = Counter(votes)
            top_label, top_count = cnt.most_common(1)[0]
            regime = top_label
            score = top_count / len(votes)
            return {"found": True, "regime": regime, "score": float(score), "neighbor_ids": ids, "distances": distances[0].tolist()}
        else:
            return {"found": False, "reason": "no_labels_on_neighbors"}

    def get_index_metrics(self) -> Dict[str, Any]:
        """
        Return basic metrics about the ANN index and mapping (file size, num_elements, dim).
        Non-fatal: returns info available.
        """
        metrics = {"index_loaded": False, "index_path": INDEX_PATH, "mapping_path": MAPPING_PATH, "num_elements": 0, "dim": self.dim}
        try:
            if self.index is None:
                # try load if exists
                if os.path.exists(INDEX_PATH):
                    try:
                        self.load_index()
                    except Exception as e:
                        logger.warning("RegimeService.get_index_metrics: failed to load index: {}", e)
            if self.index is not None:
                metrics["index_loaded"] = True
                try:
                    metrics["num_elements"] = int(self.index.get_current_count())
                except Exception:
                    # fallback: infer from mapping
                    metrics["num_elements"] = len(self.idx_to_id) if self.idx_to_id else 0
                metrics["dim"] = int(self.dim) if self.dim is not None else None
            # file sizes
            try:
                if os.path.exists(INDEX_PATH):
                    metrics["index_file_size_bytes"] = os.path.getsize(INDEX_PATH)
                if os.path.exists(MAPPING_PATH):
                    metrics["mapping_file_size_bytes"] = os.path.getsize(MAPPING_PATH)
            except Exception:
                pass
        except Exception as e:
            logger.exception("RegimeService.get_index_metrics failed: {}", e)
        return metrics

    def rebuild_async(self, n_clusters: int = 8, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Trigger asynchronous rebuild of clusters and index.
        Returns a ticket dict with status info; actual work is scheduled thread that runs fit_clusters_and_index.
        """
        import threading

        result = {"started": False, "message": "", "task": None}
        def _worker():
            try:
                logger.info("RegimeService.rebuild_async: starting rebuild n_clusters={} limit={}", n_clusters, limit)
                self.fit_clusters_and_index(n_clusters=n_clusters, limit=limit)
                logger.info("RegimeService.rebuild_async: rebuild completed")
            except Exception as e:
                logger.exception("RegimeService.rebuild_async: rebuild failed: {}", e)

        try:
            t = threading.Thread(target=_worker, name="RegimeRebuildWorker", daemon=True)
            t.start()
            result["started"] = True
            result["task"] = t.name
            result["message"] = "Rebuild started"
        except Exception as e:
            result["message"] = str(e)
        return result
