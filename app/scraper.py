"""Web scraper — fetch, parse, chunk, embed, and store game content."""

import asyncio
import re

import httpx
from bs4 import BeautifulSoup

from app import database as db
from app.embedding import encode_batch
from config import settings

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


def _is_heading(line: str) -> str | None:
    """Return the heading text if the line is a Markdown heading, else None."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    m = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", stripped)
    return m.group(2).strip() if m else None


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Each chunk is prefixed with its current heading path (e.g. "【重要物品获取方法 > 梦之钉】")
    so the chunk is self-contained for retrieval — a chunk about "位置: 静息之地"
    under a "梦之钉获取方法" heading will contain the keyword "梦之钉".
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    lines = text.split("\n")
    # Build a list of (line, heading_path_at_that_line)
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    annotated: list[tuple[str, str]] = []  # (line, heading_path)
    for line in lines:
        heading = _is_heading(line)
        if heading:
            level = len(line.lstrip().split()[0])  # count of '#'
            # Pop headings at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))
            # Don't include the heading line itself as content (it becomes context)
            continue
        path = " > ".join(h[1] for h in heading_stack)
        annotated.append((line, path))

    # Group into paragraphs (consecutive non-empty lines with same heading path)
    paragraphs: list[tuple[str, str]] = []  # (text, heading_path)
    for line, path in annotated:
        if not line.strip():
            continue
        if paragraphs and paragraphs[-1][1] == path:
            paragraphs[-1] = (paragraphs[-1][0] + "\n" + line, path)
        else:
            paragraphs.append((line, path))

    chunks: list[str] = []
    current_chunk = ""
    current_path = ""
    for para_text, path in paragraphs:
        candidate = para_text
        # Prefix with heading context so the chunk is self-contained
        prefix = f"【{path}】\n" if path else ""
        item = prefix + candidate
        if len(current_chunk) + len(candidate) <= chunk_size:
            if current_chunk and current_path == path:
                current_chunk += "\n" + candidate
            else:
                if current_chunk.strip():
                    chunks.append((current_path + "\n" + current_chunk if current_path else current_chunk).strip())
                current_chunk = candidate
                current_path = path
        else:
            if current_chunk.strip():
                chunks.append((current_path + "\n" + current_chunk if current_path else current_chunk).strip())
            if len(candidate) > chunk_size:
                # Paragraph too long — split by character window, keep heading prefix
                for i in range(0, len(candidate), chunk_size - overlap):
                    piece = candidate[i:i + chunk_size].strip()
                    if piece:
                        chunks.append((prefix + piece).strip())
                current_chunk = ""
                current_path = ""
            else:
                current_chunk = candidate
                current_path = path
    if current_chunk.strip():
        chunks.append((current_path + "\n" + current_chunk if current_path else current_chunk).strip())
    return chunks


def parse_chunk_sentences(chunk_text: str) -> list[dict]:
    """
    解析 chunk 的标题前缀，分割为单独的语义句子，并为每个句子贴上对应的语义标签 (Tag)。
    """
    # 匹配开头的【Heading Path】\n
    m = re.match(r"^【(.*?)】\n", chunk_text)
    if m:
        path = m.group(1)
        tag = path.split(">")[-1].strip() if ">" in path else path.strip()
        body = chunk_text[m.end():]
    else:
        # 兼容没有【】括号但第一行是短的标题路径的情况
        lines = chunk_text.split("\n")
        first_line = lines[0].strip()
        if len(first_line) <= 60 and not any(p in first_line for p in ["。", "！", "？", "!", "?", "；", ";"]):
            path = first_line
            path_clean = path.strip("【】")
            tag = path_clean.split(">")[-1].strip() if ">" in path_clean else path_clean.strip()
            body = "\n".join(lines[1:])
        else:
            tag = "通用百科"
            body = chunk_text

    # 使用标点符号和换行符切割为语义完整的单句
    sentence_split_rx = re.compile(r'([^。！？!?\n]+[。！？!?\n]*)')
    parts = sentence_split_rx.findall(body)

    sentences = []
    for part in parts:
        s = part.strip()
        if len(s) >= 8:  # 忽略长度过短的噪声（如列表数字等）
            sentences.append({
                "content": s,
                "tag": tag
            })
    return sentences


async def scrape_and_store(
    game_name: str,
    source_name: str,
    source_url: str,
) -> dict:
    """
    Fetch a web page, extract text, chunk it, embed, and store in the database.

    Returns:
        dict with keys: status, chunks_stored, error
    """
    try:
        # Step 1: Fetch the page
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(source_url, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()

        # Step 2: Parse HTML and extract text
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        # Extract title
        title = None
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # Get main content
        content_elem = soup.find("article") or soup.find("main") or soup.find("body")
        if content_elem:
            text = content_elem.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean text
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        if not text or len(text) < 50:
            return {"status": "failed", "chunks_stored": 0, "error": "Page content too short or empty"}

        # Step 3: Chunk the text
        chunks = chunk_text(text)
        if not chunks:
            return {"status": "failed", "chunks_stored": 0, "error": "No valid chunks produced"}

        # Step 4: Batch-embed all chunks
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, encode_batch, chunks)

        # Step 5: Delete old documents from the same URL
        db.delete_documents_by_url(source_url)

        # Step 6: Store all chunks along with semantic chunks
        await db.store_documents_with_semantic_chunks(
            game_name=game_name,
            chunks=chunks,
            embeddings=embeddings,
            title=title,
            url=source_url,
            source_name=source_name,
        )

        return {"status": "completed", "chunks_stored": len(chunks), "error": None}

    except httpx.HTTPError as e:
        return {"status": "failed", "chunks_stored": 0, "error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"status": "failed", "chunks_stored": 0, "error": str(e)}
