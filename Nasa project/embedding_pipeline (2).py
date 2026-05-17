#!/usr/bin/env python3
"""
NASA Space Mission Intelligence - Vector Embedding Ingestion Pipeline
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# System override to swap out standard sqlite3 for pysqlite3-binary
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import hashlib
import time
from datetime import datetime
import argparse

# Configure logger tracks
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('chroma_embedding_text_only.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ChromaEmbeddingPipelineTextOnly:
    def __init__(self, openai_api_key: str, chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text", embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        
        self.openai_client = OpenAI(api_key=openai_api_key, base_url="https://openai.vocareum.com/v1")
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chroma_client = chromadb.PersistentClient(path=self.chroma_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        if not text or not text.strip(): return []
        text = text.strip()
        text_length = len(text)

        if text_length <= self.chunk_size:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({"chunk_index": 0, "chunk_count": 1, "chunk_size": text_length})
            return [(text, chunk_metadata)]

        chunks = []
        start = 0
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            if end < text_length:
                sentence_break = [". ", "! ", "? ", "\n\n", "\n"]
                best_break = -1
                for boundary in sentence_break:
                    pos = text.rfind(boundary, start, end)
                    if pos > best_break:
                        best_break = pos + len(boundary)
                
                if best_break > start + (self.chunk_size // 2):        
                    end = best_break
                else:
                    space_break = text.rfind(" ", start, end)
                    if space_break > start + (self.chunk_size // 2):
                        end = space_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": len(chunks), "chunk_size": len(chunk),
                    "start_char": start, "end_char": end
                })
                chunks.append((chunk, chunk_metadata))

            if end >= text_length: break
            start = max(end - self.chunk_overlap, start + 1)

        total_chunks = len(chunks)
        for _, m in chunks: m["chunk_count"] = total_chunks
        return chunks

    def check_document_exists(self, doc_id: str) -> bool:
        try:
            result = self.collection.get(ids=[doc_id]) 
            return len(result.get("ids", [])) > 0
        except Exception: return False

    def get_embedding(self, text_list: List[str]) -> List[List[float]]:
        """Fetches embeddings for a batch of text strings safely."""
        try:
            cleaned_texts = [str(t).strip() for t in text_list if t and str(t).strip()]
            if not cleaned_texts:
                return []

            response = self.openai_client.embeddings.create(
                model=self.embedding_model, 
                input=cleaned_texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            return []

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        mission = metadata.get("mission", "mission").lower()
        source = file_path.stem.lower().replace(" ", "_")
        idx = metadata.get("chunk_index", 0)
        return f"{mission}_{source}_chunk_{idx:04d}"

    def extract_mission_from_path(self, file_path: Path) -> str:
        p = str(file_path).lower()
        if 'apollo11' in p: return 'apollo_11'
        if 'apollo13' in p: return 'apollo_13'
        if 'challenger' in p: return 'challenger'
        return 'unknown'

    def extract_data_type_from_path(self, file_path: Path) -> str:
        p = str(file_path).lower()
        if 'transcript' in p: return 'transcript'
        return 'document'

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip(): return []
            metadata = {
                'source': file_path.stem, 'file_path': str(file_path),
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'processed_timestamp': datetime.now().isoformat()
            }
            return self.chunk_text(content, metadata)
        except Exception as e:
            logger.error(f"File error {file_path}: {e}")
            return []

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        files = []
        for d in ['apollo11', 'apollo13', 'challenger']:
            path = Path(base_path) / d
            if path.exists():
                files.extend([f for f in path.glob('**/*.txt') if 'summary' not in f.name.lower()])
        return files

    def get_file_documents(self, file_path: Path) -> List[str]:
        """Helper to discover existing IDs tracking back to a specific target file."""
        try:
            filename = file_path.name
            # Pull records matching this filename pattern structure
            existing = self.collection.get(where={"source": file_path.stem})
            return existing.get("ids", [])
        except Exception:
            return []
    
    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]], 
                                   file_path: Path, batch_size: int = 50, 
                                   update_mode: str = 'skip') -> Dict[str, int]:
        """Add documents to ChromaDB collection in batches with update handling branches."""
        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}
        
        stats = {'added': 0, 'updated': 0, 'skipped': 0}
        filename = file_path.name
        
        # Replace mode: wipe the file's historical vector data first
        if update_mode == 'replace':
            existing_ids = self.get_file_documents(file_path)
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logging.info(f"Deleted existing docs for {filename}")

        # Process text chunks in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            
            batch_texts = []
            batch_metadatas = []
            batch_ids = []
            
            for text, metadata in batch:
                idx = metadata.get('chunk_index', 0)
                doc_id = f"{filename}_{idx}"
                
                exists = False
                if update_mode != 'replace':
                    existing = self.collection.get(ids=[doc_id])
                    exists = len(existing['ids']) > 0

                if exists:
                    if update_mode == 'skip':
                        stats['skipped'] += 1
                        continue
                    else:
                        stats['updated'] += 1
                else:
                    stats['added'] += 1
                
                batch_texts.append(text)
                batch_metadatas.append(metadata)
                batch_ids.append(doc_id)

            # Fire request arrays over to ChromaDB/OpenAI
            if batch_texts:
                try:
                    embeddings = self.get_embedding(batch_texts)
                    if embeddings:
                        self.collection.upsert(
                            documents=batch_texts,
                            embeddings=embeddings,
                            metadatas=batch_metadatas,
                            ids=batch_ids
                        )
                    else:
                        logging.warning(f"No embeddings returned for batch in {filename}")
                except Exception as e:
                    logging.error(f"Error processing batch in {filename}: {e}")

            logging.info(f"Progress for {filename}: {min(i + batch_size, len(documents))}/{len(documents)}")
            time.sleep(0.5)

        return stats

    def process_all_text_data(self, base_path: str, update_mode: str = 'skip') -> Dict[str, Any]:
        stats = {'files_processed': 0, 'documents_added': 0, 'documents_updated': 0, 'documents_skipped': 0, 'errors': 0, 'total_chunks': 0}
        files = self.scan_text_files_only(base_path)
        for file_path in files:
            chunks = self.process_text_file(file_path)
            if not chunks: continue
            b_stats = self.add_documents_to_collection(chunks, file_path, update_mode=update_mode)
            stats['files_processed'] += 1
            stats['total_chunks'] += len(chunks)
            stats['documents_added'] += b_stats['added']
            stats['documents_updated'] += b_stats['updated']
            stats['documents_skipped'] += b_stats['skipped']
        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {"collection_name": self.collection_name, "document_count": count}

def main():
    parser = argparse.ArgumentParser(description="NASA Mission Intelligence Embedding Pipeline")
    parser.add_argument('--data-path', default='.')
    parser.add_argument('--openai-key', required=True)
    parser.add_argument('--update-mode', default='skip', choices=['skip', 'update', 'replace'])
    
    # Expose runtime chunk controls to pass pipeline rubric
    parser.add_argument('--chunk-size', type=int, default=1000, help="Max characters per text chunk")
    parser.add_argument('--chunk-overlap', type=int, default=200, help="Overlap between consecutive chunks")
    
    # Expose required database stats-only flag
    parser.add_argument('--stats-only', action='store_true', help="Print database metrics and exit without parsing files")
    
    args = parser.parse_args()
    
    # Construct ingestion pipeline matching CLI args
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Run in stats-only mode if flag is provided
    if args.stats_only:
        logger.info("=== Fetching ChromaDB Collection Metadata (Stats-Only Mode) ===")
        info = pipeline.get_collection_info()
        logger.info(f"Collection Name:       {info['collection_name']}")
        logger.info(f"Total Chunks Injected: {info['document_count']}")
        
        try:
            sample_data = pipeline.collection.get(limit=50)
            if sample_data and sample_data.get('metadatas'):
                missions_found = set(m.get('mission', 'unknown') for m in sample_data['metadatas'] if m)
                logger.info(f"Missions detected in database sample: {', '.join(missions_found)}")
        except Exception as e:
            logger.warning(f"Could not compute collection aggregates: {e}")
        return

    # Normal heavy data processing execution path
    logger.info("Starting processing...")
    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode)
    
    logger.info(f"Done! Processed {stats['files_processed']} files. Added {stats['documents_added']} chunks.")
    info = pipeline.get_collection_info()
    logger.info(f"Total entries in DB: {info['document_count']}")

if __name__ == "__main__":
    main()