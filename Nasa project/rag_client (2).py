__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb  # Now this will work!
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

from pathlib import Path
from typing import Dict
import chromadb

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Setup the embedding function to match your database
def get_embedding_function(api_key):
    return OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small", # Matches your 1536 dim database
        api_base="https://openai.vocareum.com/v1" # Crucial for Udacity
    )


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for likely ChromaDB persistence directories
    chroma_dirs = [
        path for path in current_dir.iterdir()
        if path.is_dir() and "chroma" in path.name.lower()
    ]

    # Loop through each discovered directory
    for db_dir in chroma_dirs:
        try:
            # Initialize database client with directory path
            client = chromadb.PersistentClient(path=str(db_dir))

            # Retrieve list of available collections
            collections = client.list_collections()

            # If the directory is valid but has no collections, still show it
            if not collections:
                backend_key = f"{db_dir.name}__empty"
                backends[backend_key] = {
                    "directory": str(db_dir),
                    "collection_name": "",
                    "display_name": f"{db_dir.name} (no collections found)",
                    "document_count": "0"
                }
                continue

            # Loop through each collection found
            for collection in collections:
                collection_name = collection.name

                # Create unique identifier key combining directory and collection names
                backend_key = f"{db_dir.name}__{collection_name}"

                # Try to get document count
                try:
                    document_count = str(collection.count())
                except Exception:
                    document_count = "unknown"

                # Build information dictionary
                backends[backend_key] = {
                    "directory": str(db_dir),
                    "collection_name": collection_name,
                    "display_name": f"{db_dir.name} / {collection_name} ({document_count} docs)",
                    "document_count": document_count
                }

        except Exception as e:
            # Handle connection or access errors gracefully
            error_text = str(e)
            if len(error_text) > 50:
                error_text = error_text[:50] + "..."

            backend_key = f"{db_dir.name}__error"
            backends[backend_key] = {
                "directory": str(db_dir),
                "collection_name": "",
                "display_name": f"{db_dir.name} (error: {error_text})",
                "document_count": "unknown"
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
     
        client = chromadb.PersistentClient(path=chroma_dir)

        collection = client.get_collection(name=collection_name)

        return collection, True, ""

    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return None, False, str(e)
def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    try:
        # Initialize filter variable to None (represents no filtering)
        where_filter = None

        # Apply mission filter if provided and not "all"
        if mission_filter and mission_filter.lower() != "all":
            where_filter = {"mission": mission_filter}

        # Execute database query
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        # Return query results to caller
        return results

    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return None

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # Initialize list with header
    context_parts = ["Retrieved NASA Mission Context:\n"]

    # Loop through documents and metadata
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        # Extract metadata with fallbacks
        mission = meta.get("mission", "unknown_mission")
        source = meta.get("source", "unknown_source")
        category = meta.get("category", "general")

        # Clean formatting
        mission_clean = mission.replace("_", " ").title()
        category_clean = category.replace("_", " ").title()

        # Create header
        header = f"[Source {i}] Mission: {mission_clean} | Source: {source} | Category: {category_clean}"
        context_parts.append(header)

        # Truncate document if too long
        max_length = 500
        if len(doc) > max_length:
            doc = doc[:max_length].strip() + "..."

        context_parts.append(doc)
        context_parts.append("")  # spacing

    # Join everything
    return "\n".join(context_parts)