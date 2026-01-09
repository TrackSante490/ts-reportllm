import os
from pathlib import Path
import frontmatter
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
KB_DIR = PROJECT_ROOT / "knowledge_base"

def load_markdown_kb():
    print("KB_DIR:", KB_DIR)  # DEBUG
    documents = []
    md_count = 0

    for root, _, files in os.walk(KB_DIR):
        for file in files:
            if not file.endswith(".md"):
                continue

            md_count += 1
            path = Path(root) / file

            post = frontmatter.load(str(path))
            text = (post.content or "").strip()

            # DEBUG: show which files are empty after parsing
            if not text:
                print(f"Skipping empty content: {path}")
                continue

            metadata = post.metadata if isinstance(post.metadata, dict) else {}
            metadata = {**metadata, "source": file, "path": str(path)}

            documents.append(
                Document(
                    page_content=text,   # use cleaned text
                    metadata=metadata
                )
            )

    print(f"Found {md_count} markdown files, loaded {len(documents)} docs")  # DEBUG
    return documents
