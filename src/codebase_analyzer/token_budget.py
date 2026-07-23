from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    file_path: str
    part: int
    content: str
    token_count: int


class TokenBudgeter:
    def __init__(self, model: str) -> None:
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])

    def chunk_file(self, file_path: str, content: str, max_tokens: int) -> list[CodeChunk]:
        if self.count(content) <= max_tokens:
            return [
                CodeChunk(
                    chunk_id=f"{file_path}::part-1",
                    file_path=file_path,
                    part=1,
                    content=content,
                    token_count=self.count(content),
                )
            ]

        chunks: list[CodeChunk] = []
        current_lines: list[str] = []
        current_tokens = 0
        part = 1
        for line in content.splitlines(keepends=True):
            line_tokens = self.count(line)
            if line_tokens > max_tokens:
                if current_lines:
                    chunks.append(self._chunk(file_path, part, "".join(current_lines)))
                    part += 1
                    current_lines = []
                    current_tokens = 0
                encoded = self.encoding.encode(line)
                for offset in range(0, len(encoded), max_tokens):
                    segment = self.encoding.decode(encoded[offset : offset + max_tokens])
                    chunks.append(self._chunk(file_path, part, segment))
                    part += 1
                continue
            if current_lines and current_tokens + line_tokens > max_tokens:
                chunks.append(self._chunk(file_path, part, "".join(current_lines)))
                part += 1
                current_lines = []
                current_tokens = 0
            current_lines.append(line)
            current_tokens += line_tokens
        if current_lines:
            chunks.append(self._chunk(file_path, part, "".join(current_lines)))
        return chunks

    def pack_batches(
        self,
        chunks: list[CodeChunk],
        usable_tokens: int,
        extra_tokens_by_chunk: dict[str, int] | None = None,
        max_chunks_per_batch: int | None = None,
    ) -> list[list[CodeChunk]]:
        extra_tokens_by_chunk = extra_tokens_by_chunk or {}
        batches: list[list[CodeChunk]] = []
        current: list[CodeChunk] = []
        current_tokens = 0
        for chunk in chunks:
            effective_tokens = chunk.token_count + extra_tokens_by_chunk.get(chunk.chunk_id, 0)
            if effective_tokens > usable_tokens:
                raise ValueError(f"Chunk {chunk.chunk_id} exceeds usable batch budget")
            batch_is_full = (
                max_chunks_per_batch is not None and len(current) >= max_chunks_per_batch
            )
            if current and (current_tokens + effective_tokens > usable_tokens or batch_is_full):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(chunk)
            current_tokens += effective_tokens
        if current:
            batches.append(current)
        return batches

    def _chunk(self, file_path: str, part: int, content: str) -> CodeChunk:
        return CodeChunk(
            chunk_id=f"{file_path}::part-{part}",
            file_path=file_path,
            part=part,
            content=content,
            token_count=self.count(content),
        )
