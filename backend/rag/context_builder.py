"""Assemble grounded answer context, citations, and media cards."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import tiktoken

from retrieval.hybrid_search import QueryBundle, RetrievalCandidate
from retrieval.reranker import EvidenceSelection


@dataclass
class ContextPack:
    """Final context payload for the answer layer."""

    system_prompt: str
    user_prompt: str
    citations: list[dict[str, Any]]
    images: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    enough_evidence: bool
    debug: dict[str, Any] = field(default_factory=dict)


class ContextBuilder:
    """Build stable citation packs and prompts from reranked evidence."""

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "define", "do", "explain", "for", "from",
        "give", "how", "i", "in", "is", "it", "me", "of", "on", "or", "show", "tell", "that", "the", "this",
        "to", "what", "where", "which", "who", "why", "with", "you", "your",
    }

    def __init__(self, token_budget: int = 2800) -> None:
        self.token_budget = int(token_budget)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def build(self, query_bundle: QueryBundle, evidence: EvidenceSelection) -> ContextPack:
        query_keywords = self._keywords(query_bundle.query)
        citation_candidates = [
            candidate
            for candidate in evidence.citation_candidates
            if self._should_include_citation(query_bundle, candidate, query_keywords)
        ]
        citations: list[dict[str, Any]] = []
        citation_id_map: dict[str, str] = {}

        for index, candidate in enumerate(citation_candidates, start=1):
            citation_id = f"C{index}"
            citation_id_map[candidate.candidate_id] = citation_id
            citations.append(self._make_citation(candidate, citation_id))

        image_cards: list[dict[str, Any]] = []
        for index, candidate in enumerate(evidence.image_candidates, start=1):
            card = self._make_image_card(
                query_bundle,
                candidate,
                card_id=f"F{index}",
                citation_id_map=citation_id_map,
                query_keywords=query_keywords,
            )
            if card:
                image_cards.append(card)

        table_cards: list[dict[str, Any]] = []
        for index, candidate in enumerate(evidence.table_candidates, start=1):
            card = self._make_table_card(query_bundle, candidate, card_id=f"T{index}", citation_id_map=citation_id_map)
            if card:
                table_cards.append(card)

        enough_evidence = self._enough_evidence(query_bundle, evidence, citations, image_cards, table_cards)
        system_prompt, user_prompt, prompt_debug = self._build_prompt(
            query_bundle=query_bundle,
            citations=citations,
            images=image_cards,
            tables=table_cards,
        )

        debug = {
            "token_budget": self.token_budget,
            "prompt_tokens": prompt_debug["prompt_tokens"],
            "prompt_sections": prompt_debug["prompt_sections"],
            "citation_ids": [citation["id"] for citation in citations],
            "image_ids": [image["id"] for image in image_cards],
            "table_ids": [table["id"] for table in table_cards],
            "query_keywords": sorted(query_keywords),
        }
        return ContextPack(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            citations=citations,
            images=image_cards,
            tables=table_cards,
            enough_evidence=enough_evidence,
            debug=debug,
        )

    def _build_prompt(
        self,
        *,
        query_bundle: QueryBundle,
        citations: list[dict[str, Any]],
        images: list[dict[str, Any]],
        tables: list[dict[str, Any]],
    ) -> tuple[str, str, dict[str, Any]]:
        system_prompt = (
            "You answer questions using only the provided evidence.\n"
            "Rules:\n"
            "1. Cite factual claims with citation IDs like C1, C2.\n"
            "2. Mention figures only if figure IDs like F1 exist.\n"
            "3. Mention tables only if table IDs like T1 exist.\n"
            "4. Preserve Sanskrit and other source scripts exactly as given.\n"
            "5. If the evidence is insufficient, say so clearly instead of guessing.\n"
            "6. Keep the answer concise but useful."
        )

        sections: list[tuple[str, str]] = [
            ("Question", query_bundle.query),
            ("Intent", query_bundle.intent),
        ]

        citation_text = "\n\n".join(self._format_citation_prompt_block(citation) for citation in citations)
        if citation_text:
            sections.append(("Text Evidence", citation_text))

        image_text = "\n\n".join(self._format_image_prompt_block(image) for image in images)
        if image_text:
            sections.append(("Figure Evidence", image_text))

        table_text = "\n\n".join(self._format_table_prompt_block(table) for table in tables)
        if table_text:
            sections.append(("Table Evidence", table_text))

        prompt_sections: list[tuple[str, str]] = []
        used_tokens = 0
        for title, content in sections:
            block = f"{title}:\n{content}".strip()
            block_tokens = self._token_count(block)
            if used_tokens + block_tokens <= self.token_budget:
                prompt_sections.append((title, content))
                used_tokens += block_tokens
                continue

            truncated = self._truncate_to_tokens(content, max(64, self.token_budget - used_tokens - self._token_count(title) - 4))
            if truncated:
                prompt_sections.append((title, truncated))
                used_tokens = self.token_budget
            break

        user_prompt = "\n\n".join(f"{title}:\n{content}" for title, content in prompt_sections if content)
        return system_prompt, user_prompt, {"prompt_tokens": used_tokens, "prompt_sections": [title for title, _ in prompt_sections]}

    def _make_citation(self, candidate: RetrievalCandidate, citation_id: str) -> dict[str, Any]:
        return {
            "id": citation_id,
            "kind": candidate.chunk_type or candidate.kind,
            "doc_id": candidate.doc_id,
            "source_file": candidate.source_file,
            "page_numbers": list(candidate.page_numbers),
            "section_path": list(candidate.section_path),
            "snippet": candidate.snippet,
        }

    def _make_image_card(
        self,
        query_bundle: QueryBundle,
        candidate: RetrievalCandidate,
        *,
        card_id: str,
        citation_id_map: dict[str, str],
        query_keywords: set[str],
    ) -> dict[str, Any] | None:
        if not candidate.image_url:
            return None
        if not self._should_include_image(query_bundle, candidate, query_keywords):
            return None
        citation_ids = [citation_id_map[candidate_id] for candidate_id in candidate.linked_ids if candidate_id in citation_id_map]
        return {
            "id": card_id,
            "image_id": candidate.candidate_id,
            "page_number": candidate.page_numbers[0] if candidate.page_numbers else None,
            "caption": candidate.caption or "",
            "labels": list(candidate.labels),
            "image_url": candidate.image_url,
            "cloudinary_public_id": candidate.cloudinary_public_id,
            "source_file": candidate.source_file,
            "citation_ids": citation_ids,
        }

    def _make_table_card(
        self,
        query_bundle: QueryBundle,
        candidate: RetrievalCandidate,
        *,
        card_id: str,
        citation_id_map: dict[str, str],
    ) -> dict[str, Any] | None:
        if not self._should_display_table(query_bundle, candidate):
            return None
        citation_id = citation_id_map.get(candidate.candidate_id)
        return {
            "id": card_id,
            "table_id": candidate.payload.get("table_id") or candidate.candidate_id,
            "page_numbers": list(candidate.page_numbers),
            "table_caption": candidate.payload.get("table_caption"),
            "table_markdown": candidate.table_markdown,
            "table_rows": candidate.table_rows,
            "source_file": candidate.source_file,
            "citation_ids": [citation_id] if citation_id else [],
        }

    def _should_display_table(self, query_bundle: QueryBundle, candidate: RetrievalCandidate) -> bool:
        if not candidate.is_table:
            return False
        if not candidate.table_markdown and not candidate.table_rows:
            return False
        if candidate.score < -0.25:
            return False
        row_count = len(candidate.table_rows or [])
        if query_bundle.is_table:
            return row_count >= 1 or bool(candidate.table_markdown)
        return row_count >= 3 and candidate.score >= 0.15

    def _enough_evidence(
        self,
        query_bundle: QueryBundle,
        evidence: EvidenceSelection,
        citations: list[dict[str, Any]],
        images: list[dict[str, Any]],
        tables: list[dict[str, Any]],
    ) -> bool:
        if citations:
            top_score = float(evidence.citation_candidates[0].score)
            if top_score >= 0.1:
                return True
        if query_bundle.is_visual:
            top_image_score = float(evidence.image_candidates[0].score) if evidence.image_candidates else -999.0
            return bool(images) and (bool(citations) or top_image_score >= 0.35)
        if query_bundle.is_table:
            top_table_score = float(evidence.table_candidates[0].score) if evidence.table_candidates else -999.0
            return bool(tables) and (bool(citations) or top_table_score >= 0.2)
        return False

    @staticmethod
    def _format_citation_prompt_block(citation: dict[str, Any]) -> str:
        section = " > ".join(citation.get("section_path") or [])
        section_line = f"Section: {section}\n" if section else ""
        return (
            f"[{citation['id']}] Source: {citation['source_file']} p{citation['page_numbers']}\n"
            f"{section_line}"
            f"{citation['snippet']}"
        ).strip()

    @staticmethod
    def _format_image_prompt_block(image: dict[str, Any]) -> str:
        labels = ", ".join(image.get("labels") or [])
        labels_line = f"\nLabels: {labels}" if labels else ""
        citation_line = f"\nLinked citations: {', '.join(image.get('citation_ids') or [])}" if image.get("citation_ids") else ""
        return (
            f"[{image['id']}] Source: {image['source_file']} page {image['page_number']}\n"
            f"Caption: {image.get('caption') or ''}{labels_line}{citation_line}"
        ).strip()

    @staticmethod
    def _format_table_prompt_block(table: dict[str, Any]) -> str:
        citation_line = f"\nLinked citations: {', '.join(table.get('citation_ids') or [])}" if table.get("citation_ids") else ""
        return (
            f"[{table['id']}] Source: {table['source_file']} p{table['page_numbers']}\n"
            f"Caption: {table.get('table_caption') or ''}{citation_line}\n"
            f"{table.get('table_markdown') or ''}"
        ).strip()

    def _truncate_to_tokens(self, text: str, budget: int) -> str:
        if budget <= 0:
            return ""
        tokens = self.encoding.encode(text or "")
        if len(tokens) <= budget:
            return text
        trimmed = self.encoding.decode(tokens[:budget]).strip()
        return trimmed.rstrip() + " ..."

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))

    def _should_include_citation(
        self,
        query_bundle: QueryBundle,
        candidate: RetrievalCandidate,
        query_keywords: set[str],
    ) -> bool:
        if candidate.score >= 0.45:
            return True
        overlap = self._keyword_overlap(query_keywords, candidate)
        if query_bundle.is_table and candidate.is_table:
            return bool(overlap) and candidate.score >= -0.1
        if query_bundle.is_visual:
            return bool(overlap) and candidate.score >= -0.1
        return bool(overlap) and candidate.score >= 0.05

    def _should_include_image(
        self,
        query_bundle: QueryBundle,
        candidate: RetrievalCandidate,
        query_keywords: set[str],
    ) -> bool:
        overlap = self._keyword_overlap(query_keywords, candidate)
        if query_bundle.is_visual:
            return bool(overlap) or candidate.score >= 0.4 or bool(candidate.linked_ids)
        return bool(overlap) and candidate.score >= 0.2

    def _keyword_overlap(self, query_keywords: set[str], candidate: RetrievalCandidate) -> set[str]:
        if not query_keywords:
            return set()
        candidate_text = " ".join(
            part
            for part in [
                candidate.heading_text or "",
                candidate.text or "",
                candidate.caption or "",
                " ".join(candidate.labels or []),
                candidate.table_markdown or "",
                candidate.snippet or "",
            ]
            if part
        )
        return query_keywords.intersection(self._keywords(candidate_text))

    def _keywords(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\w+", str(text or "").lower())
            if len(token) >= 3 and token not in self.STOPWORDS and not token.isdigit()
        }
