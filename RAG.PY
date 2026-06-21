from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fitz
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = PROJECT_DIR / "LEY 769 DE 2002.pdf"
DEFAULT_INDEX_PATH = PROJECT_DIR / "vectorstore" / "ley_769_index.json"

ASSISTANT_NAME = "LexTransito"
DEFAULT_CHAT_MODEL = "gpt-5.5"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 1400
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_TOP_K = 3
DEFAULT_RETRIEVER = "auto"
DEFAULT_GENERATION_MODE = "auto"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "all-minilm"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


SYSTEM_PROMPT = f"""
Eres {ASSISTANT_NAME}, asistente legal especializado en la Ley 769 de 2002,
Codigo Nacional de Transito Terrestre de Colombia.

Reglas obligatorias:
1. Responde unicamente con informacion del CONTEXTO RECUPERADO.
2. Si la respuesta no esta en el contexto, dilo explicitamente y no inventes.
3. No menciones numeros de articulos; cita solo paginas.
4. No inventes sanciones, excepciones ni paginas.
5. No des asesoria legal personalizada; explica lo que dice el documento.
6. Cita al final las paginas del documento consultadas con el formato:
   Paginas consultadas: p. X, p. Y.
7. Si el usuario pide algo fuera del documento, responde que no aparece en
   el contexto recuperado.
""".strip()


@dataclass
class Chunk:
    chunk_id: str
    page: int
    text: str
    embedding: list[float] | None = None


def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env without requiring extra packages."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def require_api_key() -> None:
    if OpenAI is None:
        raise RuntimeError("Falta instalar el SDK de OpenAI. Ejecuta `pip install openai`.")

    if os.getenv("OPENAI_API_KEY"):
        return

    raise RuntimeError(
        "Falta OPENAI_API_KEY. Crea un archivo .env con OPENAI_API_KEY=tu_clave "
        "o define la variable de entorno antes de ejecutar el sistema."
    )


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and OpenAI is not None


def is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "insufficient_quota" in text or "exceeded your current quota" in text


def is_ollama_available(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size debe ser mayor que overlap.")

    text = clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        candidate = text[start:end]

        if end < len(text):
            split_at = max(candidate.rfind("\n\n"), candidate.rfind(". "), candidate.rfind("; "))
            if split_at > chunk_size * 0.55:
                end = start + split_at + 1
                candidate = text[start:end]

        candidate = candidate.strip()
        if candidate:
            chunks.append(candidate)

        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def load_pdf_chunks(pdf_path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"No se encontro el PDF: {pdf_path}")

    chunks: list[Chunk] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_text = clean_text(page.get_text("text"))
            for local_index, chunk_text in enumerate(split_text(page_text, chunk_size, overlap), start=1):
                chunks.append(
                    Chunk(
                        chunk_id=f"p{page_index:03d}_c{local_index:02d}",
                        page=page_index,
                        text=chunk_text,
                    )
                )
    return chunks


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_texts(client: OpenAI, texts: list[str], model: str, batch_size: int = 64) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for batch in batched(texts, batch_size):
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def ollama_embed_texts(texts: list[str], model: str, base_url: str, batch_size: int = 32) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for batch in batched(texts, batch_size):
        payload = {"model": model, "input": batch}
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama respondio con error HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "No pude conectar con Ollama para generar embeddings. "
                "Verifica que Ollama este abierto en http://localhost:11434."
            ) from exc

        batch_embeddings = result.get("embeddings")
        if not isinstance(batch_embeddings, list):
            raise RuntimeError("Ollama no devolvio embeddings.")
        embeddings.extend(batch_embeddings)
    return embeddings


def normalize_for_tokens(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_accents


def tokenize(text: str) -> list[str]:
    stopwords = {
        "a",
        "al",
        "ante",
        "bajo",
        "con",
        "contra",
        "de",
        "del",
        "desde",
        "durante",
        "e",
        "el",
        "en",
        "entre",
        "es",
        "esta",
        "este",
        "la",
        "las",
        "ley",
        "lo",
        "los",
        "o",
        "para",
        "por",
        "que",
        "se",
        "segun",
        "sin",
        "sobre",
        "su",
        "sus",
        "un",
        "una",
        "y",
        "769",
        "2002",
    }
    tokens = re.findall(r"[a-z0-9]{2,}", normalize_for_tokens(text))
    return [stem_token(token) for token in tokens if token not in stopwords]


def stem_token(token: str) -> str:
    replacements = (
        ("aciones", "acion"),
        ("ciones", "cion"),
        ("mente", ""),
        ("idades", "idad"),
        ("adores", "ador"),
        ("adoras", "ador"),
    )
    for suffix, replacement in replacements:
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[: -len(suffix)] + replacement
            break

    for suffix in ("ando", "iendo", "ados", "adas", "idos", "idas", "es", "s", "an", "en", "ar", "er", "ir", "a", "e", "o"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def vectorize_tokens(tokens: list[str], idf: dict[str, float]) -> tuple[dict[str, float], float]:
    counts = Counter(tokens)
    vector: dict[str, float] = {}
    for term, count in counts.items():
        if term not in idf:
            continue
        vector[term] = (1.0 + math.log(count)) * idf[term]
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    return vector, norm


def build_local_index(
    pdf_path: Path,
    index_path: Path,
    chunk_size: int,
    overlap: int,
) -> dict:
    chunks = load_pdf_chunks(pdf_path, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise RuntimeError("No se pudo extraer texto del PDF.")

    print(f"Documento cargado: {pdf_path.name}")
    print(f"Chunks creados: {len(chunks)}")
    print("Generando vectores locales TF-IDF...")

    tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_chunks:
        document_frequency.update(set(tokens))

    total_docs = len(tokenized_chunks)
    idf = {
        term: math.log((1 + total_docs) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }

    stored_chunks = []
    for chunk, tokens in zip(chunks, tokenized_chunks, strict=True):
        vector, norm = vectorize_tokens(tokens, idf)
        stored_chunks.append(
            {
                "id": chunk.chunk_id,
                "page": chunk.page,
                "text": chunk.text,
                "vector": vector,
                "norm": norm,
            }
        )

    index = {
        "metadata": {
            "document": pdf_path.name,
            "assistant_name": ASSISTANT_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retriever": "local_tfidf",
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "chunks": len(chunks),
        },
        "idf": idf,
        "chunks": stored_chunks,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"Indice local guardado en: {index_path}")
    return index


def build_ollama_index(
    pdf_path: Path,
    index_path: Path,
    embedding_model: str,
    ollama_url: str,
    chunk_size: int,
    overlap: int,
) -> dict:
    if not is_ollama_available(ollama_url):
        raise RuntimeError("Ollama no esta disponible. Abre Ollama antes de generar embeddings locales.")

    chunks = load_pdf_chunks(pdf_path, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise RuntimeError("No se pudo extraer texto del PDF.")

    print(f"Documento cargado: {pdf_path.name}")
    print(f"Chunks creados: {len(chunks)}")
    print(f"Generando embeddings locales con Ollama ({embedding_model})...")

    embeddings = ollama_embed_texts([chunk.text for chunk in chunks], model=embedding_model, base_url=ollama_url)
    tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_chunks:
        document_frequency.update(set(tokens))

    total_docs = len(tokenized_chunks)
    idf = {
        term: math.log((1 + total_docs) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }

    lexical_vectors = [vectorize_tokens(tokens, idf) for tokens in tokenized_chunks]
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk.embedding = embedding

    index = {
        "metadata": {
            "document": pdf_path.name,
            "assistant_name": ASSISTANT_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retriever": "ollama_embeddings",
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
                "chunk_overlap": overlap,
                "chunks": len(chunks),
        },
        "idf": idf,
        "chunks": [
            {
                "id": chunk.chunk_id,
                "page": chunk.page,
                "text": chunk.text,
                "embedding": chunk.embedding,
                "vector": lexical_vectors[index][0],
                "norm": lexical_vectors[index][1],
            }
            for index, chunk in enumerate(chunks)
        ],
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"Indice con embeddings locales guardado en: {index_path}")
    return index


def build_openai_index(
    pdf_path: Path,
    index_path: Path,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
) -> dict:
    require_api_key()
    client = OpenAI()

    chunks = load_pdf_chunks(pdf_path, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise RuntimeError("No se pudo extraer texto del PDF.")

    print(f"Documento cargado: {pdf_path.name}")
    print(f"Chunks creados: {len(chunks)}")
    print("Generando embeddings...")

    embeddings = embed_texts(client, [chunk.text for chunk in chunks], model=embedding_model)
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk.embedding = embedding

    index = {
        "metadata": {
            "document": pdf_path.name,
            "assistant_name": ASSISTANT_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retriever": "openai_embeddings",
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "chunks": len(chunks),
        },
        "chunks": [
            {
                "id": chunk.chunk_id,
                "page": chunk.page,
                "text": chunk.text,
                "embedding": chunk.embedding,
            }
            for chunk in chunks
        ],
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"Indice guardado en: {index_path}")
    return index


def build_index(
    pdf_path: Path,
    index_path: Path,
    embedding_model: str,
    ollama_embedding_model: str,
    ollama_url: str,
    chunk_size: int,
    overlap: int,
    retriever: str,
) -> dict:
    retriever = retriever.lower()
    if retriever in {"local", "local_tfidf", "tfidf"}:
        return build_local_index(pdf_path, index_path, chunk_size, overlap)

    if retriever in {"ollama", "ollama_embeddings"}:
        return build_ollama_index(
            pdf_path=pdf_path,
            index_path=index_path,
            embedding_model=ollama_embedding_model,
            ollama_url=ollama_url,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    if retriever in {"openai", "api", "openai_embeddings"}:
        return build_openai_index(pdf_path, index_path, embedding_model, chunk_size, overlap)

    if retriever == "auto":
        if is_ollama_available(ollama_url):
            try:
                return build_ollama_index(
                    pdf_path=pdf_path,
                    index_path=index_path,
                    embedding_model=ollama_embedding_model,
                    ollama_url=ollama_url,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            except Exception as exc:
                print(f"No se pudieron crear embeddings con Ollama ({exc}); se usara TF-IDF local.")
                return build_local_index(pdf_path, index_path, chunk_size, overlap)
        if not has_api_key():
            print("No hay OPENAI_API_KEY; se usara recuperacion local TF-IDF.")
            return build_local_index(pdf_path, index_path, chunk_size, overlap)
        try:
            return build_openai_index(pdf_path, index_path, embedding_model, chunk_size, overlap)
        except Exception as exc:
            if not is_quota_error(exc):
                raise
            print("La API no tiene cuota disponible; se usara recuperacion local TF-IDF.")
            return build_local_index(pdf_path, index_path, chunk_size, overlap)

    raise ValueError("retriever debe ser auto, ollama, openai o local.")


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise FileNotFoundError(f"No se encontro el indice vectorial: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8-sig"))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def retrieve_context_openai(
    client: OpenAI,
    index: dict,
    question: str,
    embedding_model: str,
    top_k: int,
) -> list[dict]:
    query_embedding = client.embeddings.create(model=embedding_model, input=question).data[0].embedding
    query_vector = normalize(np.array(query_embedding, dtype=np.float32))

    chunks = index["chunks"]
    matrix = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_norms[matrix_norms == 0] = 1
    normalized_matrix = matrix / matrix_norms

    similarities = normalized_matrix @ query_vector
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results: list[dict] = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(similarities[int(idx)]),
                "page": int(chunk["page"]),
                "text": chunk["text"],
                "id": chunk["id"],
            }
        )
    return results


def retrieve_context_local(index: dict, question: str, top_k: int) -> list[dict]:
    idf = index["idf"]
    query_vector, query_norm = vectorize_tokens(tokenize(question), idf)
    if query_norm == 0:
        query_norm = 1.0

    scored: list[tuple[float, dict]] = []
    for chunk in index["chunks"]:
        chunk_norm = chunk.get("norm") or 1.0
        chunk_vector = chunk.get("vector", {})
        dot_product = sum(weight * chunk_vector.get(term, 0.0) for term, weight in query_vector.items())
        score = dot_product / (query_norm * chunk_norm)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[dict] = []
    for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "page": int(chunk["page"]),
                "text": chunk["text"],
                "id": chunk["id"],
            }
        )
    return results


def retrieve_context(
    client: Any | None,
    index: dict,
    question: str,
    embedding_model: str,
    ollama_embedding_model: str,
    ollama_url: str,
    top_k: int,
) -> list[dict]:
    retriever = index.get("metadata", {}).get("retriever", "openai_embeddings")
    if retriever == "local_tfidf":
        return retrieve_context_local(index, question, top_k)

    if retriever == "ollama_embeddings":
        model = index.get("metadata", {}).get("embedding_model", ollama_embedding_model)
        query_embedding = ollama_embed_texts([question], model=model, base_url=ollama_url, batch_size=1)[0]
        query_vector = normalize(np.array(query_embedding, dtype=np.float32))

        chunks = index["chunks"]
        matrix = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norms[matrix_norms == 0] = 1
        normalized_matrix = matrix / matrix_norms

        embedding_similarities = normalized_matrix @ query_vector
        similarities = embedding_similarities

        idf = index.get("idf")
        if isinstance(idf, dict):
            lexical_query_vector, lexical_query_norm = vectorize_tokens(tokenize(question), idf)
            if lexical_query_norm == 0:
                lexical_query_norm = 1.0

            lexical_scores = []
            for chunk in chunks:
                chunk_norm = chunk.get("norm") or 1.0
                chunk_vector = chunk.get("vector", {})
                dot_product = sum(
                    weight * chunk_vector.get(term, 0.0)
                    for term, weight in lexical_query_vector.items()
                )
                lexical_scores.append(dot_product / (lexical_query_norm * chunk_norm))

            lexical_scores_array = np.array(lexical_scores, dtype=np.float32)
            similarities = (0.30 * embedding_similarities) + (0.70 * lexical_scores_array)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: list[dict] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(similarities[int(idx)]),
                    "page": int(chunk["page"]),
                    "text": chunk["text"],
                    "id": chunk["id"],
                }
            )
        return results

    if client is None:
        require_api_key()
        client = OpenAI()
    return retrieve_context_openai(client, index, question, embedding_model, top_k)


def format_context(results: list[dict]) -> str:
    formatted = []
    for item in results:
        formatted.append(
            f"[Fragmento {item['rank']} | pagina {item['page']} | "
            f"similitud {item['score']:.3f}]\n{item['text']}"
        )
    return "\n\n".join(formatted)


def format_pages(results: list[dict]) -> str:
    pages = sorted({item["page"] for item in results})
    return ", ".join(f"p. {page}" for page in pages)


def select_supporting_results(results: list[dict]) -> list[dict]:
    if not results:
        return []

    top_score = max(results[0]["score"], 0.0)
    top_page = results[0]["page"]
    selected = [
        item
        for item in results
        if item["rank"] == 1
        or item["page"] == top_page
        or (top_score > 0 and item["score"] >= max(0.08, top_score * 0.75))
    ]
    return selected[:3]


def expand_retrieval_query(question: str) -> str:
    normalized_question = normalize_for_tokens(question)
    extra_terms: list[str] = []

    asks_scope = "que regula" in normalized_question or "que rige" in normalized_question
    mentions_law = "ley" in normalized_question or "769" in normalized_question or "codigo" in normalized_question
    if asks_scope and mentions_law:
        extra_terms.append(
            "ambito aplicacion principios normas rigen territorio nacional circulacion "
            "peatones usuarios pasajeros conductores motociclistas ciclistas agentes "
            "transito vehiculos vias publicas privadas autoridades transito"
        )

    if not extra_terms:
        return question
    return f"{question} {' '.join(extra_terms)}"


def split_sentences(text: str) -> list[str]:
    flat_text = re.sub(r"\s+", " ", clean_text(text))
    sentences = re.split(r"(?<=[.!?])\s+", flat_text)
    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 40 and re.search(r"[.!?]$", sentence.strip())
    ]


def make_extractive_answer(question: str, results: list[dict]) -> str:
    query_terms = set(tokenize(question))

    def collect_candidates(source_results: list[dict]) -> list[tuple[float, int, str]]:
        collected: list[tuple[float, int, str]] = []
        for item in source_results:
            for sentence in split_sentences(item["text"]):
                sentence_terms = set(tokenize(sentence))
                if not sentence_terms:
                    continue
                overlap = len(query_terms & sentence_terms)
                if overlap == 0:
                    continue
                rank_bonus = 0.35 / max(item["rank"], 1)
                score = overlap / math.sqrt(len(sentence_terms)) + max(item["score"], 0.0) * 0.25 + rank_bonus
                collected.append((score, item["page"], sentence))
        return collected

    candidates = collect_candidates(results[:1])
    if len(candidates) < 2:
        candidates = collect_candidates(results)

    fallback_pages = format_pages(results)
    if not candidates:
        return (
            "No encontre informacion suficiente en el contexto recuperado para responder "
            f"con seguridad.\n\nPaginas consultadas: {fallback_pages}."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[int, str]] = []
    seen: set[str] = set()
    for score, page, sentence in candidates:
        normalized = normalize_for_tokens(sentence)
        if normalized in seen:
            continue
        selected.append((page, sentence))
        seen.add(normalized)
        if len(selected) == 3:
            break

    pages = ", ".join(f"p. {page}" for page in sorted({page for page, _ in selected}))
    selected.sort(key=lambda item: item[0])
    body = " ".join(sentence for _, sentence in selected)
    return f"{body}\n\nPaginas consultadas: {pages}."


def generate_with_ollama(prompt: str, model: str, base_url: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": 700,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama respondio con error HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "No pude conectar con Ollama. Abre Ollama o ejecuta `ollama serve` "
            "y verifica que responda en http://localhost:11434."
        ) from exc

    response_text = (result.get("response") or "").strip()
    if not response_text:
        raise RuntimeError("Ollama no devolvio texto.")
    return response_text


def sanitize_generated_answer(answer: str, pages: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"(?im)^Paginas consultadas:.*$", "", answer).strip()
    paragraphs = re.split(r"\n{2,}", answer)
    cleaned_paragraphs: list[str] = []

    forbidden_pattern = re.compile(r"\b(art[ií]culo|fragmento)\b", flags=re.IGNORECASE)
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        kept = [sentence.strip() for sentence in sentences if sentence.strip() and not forbidden_pattern.search(sentence)]
        if kept:
            cleaned_paragraphs.append(" ".join(kept))

    if not cleaned_paragraphs:
        cleaned_paragraphs.append("No encontre informacion suficiente en el contexto recuperado para responder con seguridad.")

    body = "\n\n".join(cleaned_paragraphs)
    return f"{body}\n\nPaginas consultadas: {pages}."


def answer_question(
    client: Any | None,
    index: dict,
    question: str,
    chat_model: str,
    embedding_model: str,
    ollama_embedding_model: str,
    top_k: int,
    recent_questions: list[str],
    generation_mode: str,
    ollama_model: str,
    ollama_url: str,
) -> tuple[str, list[dict]]:
    retrieval_query = " ".join([*recent_questions[-2:], expand_retrieval_query(question)]).strip()
    results = retrieve_context(
        client=client,
        index=index,
        question=retrieval_query,
        embedding_model=embedding_model,
        ollama_embedding_model=ollama_embedding_model,
        ollama_url=ollama_url,
        top_k=top_k,
    )
    supporting_results = select_supporting_results(results)

    generation_mode = generation_mode.lower()
    if generation_mode == "extractive":
        return make_extractive_answer(question, supporting_results), results

    context = format_context(supporting_results)
    pages = format_pages(supporting_results)
    user_prompt = f"""
CONTEXTO RECUPERADO:
{context}

PREGUNTA DEL USUARIO:
{question}

Instruccion final: responde en espanol claro, breve y fundamentado.
Debes terminar con exactamente esta linea si usaste el contexto:
Paginas consultadas: {pages}.
""".strip()
    ollama_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    if generation_mode == "ollama":
        answer = generate_with_ollama(ollama_prompt, ollama_model, ollama_url)
        return sanitize_generated_answer(answer, pages), results

    if client is None:
        if generation_mode == "api":
            require_api_key()
        elif generation_mode == "auto":
            if is_ollama_available(ollama_url):
                try:
                    answer = generate_with_ollama(ollama_prompt, ollama_model, ollama_url)
                    return sanitize_generated_answer(answer, pages), results
                except Exception as exc:
                    print(f"Aviso: Ollama no pudo generar respuesta ({exc}); usando respuesta extractiva local.")
                    return make_extractive_answer(question, supporting_results), results
            return make_extractive_answer(question, supporting_results), results
        else:
            return make_extractive_answer(question, supporting_results), results
        client = OpenAI()

    try:
        response = client.responses.create(
            model=chat_model,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
            max_output_tokens=800,
        )
        return sanitize_generated_answer(response.output_text, pages), results
    except Exception as exc:
        if generation_mode == "api" or not is_quota_error(exc):
            raise
        if is_ollama_available(ollama_url):
            print("Aviso: la API no tiene cuota; usando IA local con Ollama.")
            try:
                answer = generate_with_ollama(ollama_prompt, ollama_model, ollama_url)
                return sanitize_generated_answer(answer, pages), results
            except Exception as ollama_exc:
                print(f"Aviso: Ollama no pudo generar respuesta ({ollama_exc}); usando respuesta extractiva local.")
                return make_extractive_answer(question, supporting_results), results
        print("Aviso: la API no tiene cuota y Ollama no esta disponible; usando respuesta extractiva local.")
        return make_extractive_answer(question, supporting_results), results


def run_chat(
    index_path: Path,
    pdf_path: Path,
    chat_model: str,
    embedding_model: str,
    ollama_embedding_model: str,
    chunk_size: int,
    overlap: int,
    top_k: int,
    rebuild: bool,
    retriever: str,
    generation_mode: str,
    ollama_model: str,
    ollama_url: str,
) -> None:
    client = OpenAI() if has_api_key() else None

    if rebuild or not index_path.exists():
        print("Construyendo indice vectorial local...")
        index = build_index(
            pdf_path=pdf_path,
            index_path=index_path,
            embedding_model=embedding_model,
            ollama_embedding_model=ollama_embedding_model,
            ollama_url=ollama_url,
            chunk_size=chunk_size,
            overlap=overlap,
            retriever=retriever,
        )
    else:
        index = load_index(index_path)

    metadata = index.get("metadata", {})
    print("\nSistema RAG listo.")
    print(f"Asistente: {ASSISTANT_NAME}")
    print(f"Documento: {metadata.get('document', pdf_path.name)}")
    print(f"Chunks indexados: {metadata.get('chunks', len(index.get('chunks', [])))}")
    print(f"Recuperacion: {metadata.get('retriever', 'openai_embeddings')}")
    print(f"Generacion: {generation_mode}")
    if generation_mode in {"auto", "ollama"}:
        ollama_state = "disponible" if is_ollama_available(ollama_url) else "no disponible"
        print(f"Ollama: {ollama_state} | modelo: {ollama_model}")
    print("Escribe tu pregunta o usa 'salir' para terminar.\n")

    recent_questions: list[str] = []
    while True:
        try:
            question = input("Tu pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSesion finalizada.")
            break

        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit", "q"}:
            print("Sesion finalizada.")
            break

        try:
            answer, results = answer_question(
                client=client,
                index=index,
                question=question,
                chat_model=chat_model,
                embedding_model=embedding_model,
                ollama_embedding_model=ollama_embedding_model,
                top_k=top_k,
                recent_questions=recent_questions,
                generation_mode=generation_mode,
                ollama_model=ollama_model,
                ollama_url=ollama_url,
            )
        except Exception as exc:
            print(f"\nError al responder: {exc}\n")
            continue

        recent_questions.append(question)
        print(f"\n{ASSISTANT_NAME}: {answer}")
        print("Fragmentos recuperados:")
        for item in results:
            print(f"  - {item['id']} | pagina {item['page']} | similitud {item['score']:.3f}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema RAG sobre la Ley 769 de 2002 - Codigo Nacional de Transito."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH, help="Ruta del PDF especializado.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH, help="Ruta del indice vectorial.")
    parser.add_argument("--rebuild", action="store_true", help="Reconstruye el indice vectorial.")
    parser.add_argument("--build-index", action="store_true", help="Solo construye el indice y termina.")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAG_TOP_K", DEFAULT_TOP_K)))
    parser.add_argument(
        "--retriever",
        default=os.getenv("RAG_RETRIEVER", DEFAULT_RETRIEVER),
        choices=["auto", "ollama", "openai", "local"],
        help="Motor de recuperacion: auto, ollama, openai o local.",
    )
    parser.add_argument(
        "--generation-mode",
        default=os.getenv("RAG_GENERATION_MODE", DEFAULT_GENERATION_MODE),
        choices=["auto", "api", "ollama", "extractive"],
        help="Modo de respuesta: auto, api, ollama o extractive.",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        help="Modelo local de Ollama.",
    )
    parser.add_argument(
        "--ollama-embedding-model",
        default=os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_EMBEDDING_MODEL),
        help="Modelo local de embeddings en Ollama.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        help="URL local de Ollama.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
    )
    parser.add_argument(
        "--chat-model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL),
        help="Modelo de generacion de respuestas.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="Modelo de embeddings.",
    )
    return parser.parse_args()


def main() -> int:
    load_env_file(PROJECT_DIR / ".env")
    args = parse_args()

    try:
        if args.build_index:
            build_index(
                pdf_path=args.pdf,
                index_path=args.index,
                embedding_model=args.embedding_model,
                ollama_embedding_model=args.ollama_embedding_model,
                ollama_url=args.ollama_url,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
                retriever=args.retriever,
            )
            return 0

        run_chat(
            index_path=args.index,
            pdf_path=args.pdf,
            chat_model=args.chat_model,
            embedding_model=args.embedding_model,
            ollama_embedding_model=args.ollama_embedding_model,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            top_k=args.top_k,
            rebuild=args.rebuild,
            retriever=args.retriever,
            generation_mode=args.generation_mode,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
