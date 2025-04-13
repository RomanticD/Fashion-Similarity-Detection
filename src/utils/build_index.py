"""
Utility script to build the vector index from the database
"""

import sys
import os
from pathlib import Path

# Set up Python path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.core.vector_index import VectorIndex

if __name__ == "__main__":
    print("Building vector index from database...")
    # 创建 VectorIndex 类的实例
    index_manager = VectorIndex()
    index, ids, vectors = index_manager.rebuild_index()

    if index is not None:
        print(f"Successfully built index with {len(ids)} vectors")
    else:
        print("Failed to build index")
