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
        top_candidate = evidence.reranked_candidates[0] if evidence.reranked_candidates else None
        top_image_candidate_id = top_candidate.candidate_id if top_candidate and top_candidate.kind == "image" else None
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
                candidate,
                card_id=f"F{index}",
                citation_id_map=citation_id_map,
                is_top_ranked=(candidate.candidate_id == top_image_candidate_id),
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
            top_chunk_kind=top_candidate.kind if top_candidate else "none",
        )

        debug = {
            "token_budget": self.token_budget,
            "prompt_tokens": prompt_debug["prompt_tokens"],
            "prompt_sections": prompt_debug["prompt_sections"],
            "citation_ids": [citation["id"] for citation in citations],
            "image_ids": [image["id"] for image in image_cards],
            "table_ids": [table["id"] for table in table_cards],
            "query_keywords": sorted(query_keywords),
            "top_chunk_kind": top_candidate.kind if top_candidate else None,
            "top_chunk_id": top_candidate.candidate_id if top_candidate else None,
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
        top_chunk_kind: str,
    ) -> tuple[str, str, dict[str, Any]]:
        system_prompt = (
            "You are an expert in Ayurvedic texts with deep knowledge of Sanskrit, Hindi, and English.\n"
            "Use only the provided context to answer the question.\n"
            "Rules:\n"
            "1. Preserve original Sanskrit or Hindi wording EXACTLY when quoting. Never normalize diacritics in output. "
            "Never transliterate Sanskrit into Roman script unless explicitly asked.\n"
            "2. Provide clear explanations in the same language the question was asked.\n"
            "3. Display Sanskrit shlokas wherever needed if they are present in the evidence.\n"
            "4. If a procedure or list is present, format it as a Markdown table whenever that improves clarity.\n"
            "5. If context mentions a diagram or illustration, reference it in your answer using the supplied "
            "diagram token like [Diagram: ...].\n"
            "6. Always end your answer with a source citation in the format [Book Title, p.X] or "
            "[Book Title, Shloka X.Y] when verse numbering is available in the evidence.\n"
            "7. Cite factual claims with the short evidence IDs like C1, C2 in the body when useful, but do not invent IDs.\n"
            "8. If the answer is not in the context, say so clearly. Do not hallucinate.\n\n"
            "Few-shot examples of correct Sanskrit preservation:\n"
            "Q: What is Vata?\n"
            "A: Vata (वात) is one of the three doshas. The classical definition is: वातः पित्तं कफश्चेति त्रयो दोषाः । "
            "[Charaka Samhita, Sutra 1.57]\n\n"
            "Q: Explain Agni in digestion.\n"
            "A: Agni (अग्नि) refers to the digestive fire. The text states: जठराग्निः सर्वाग्नीनां मूलम् — meaning "
            "Jatharagni is the root of all fires in the body. [Charaka Samhita, Chikitsa 15.3]"
        )

        sections: list[tuple[str, str]] = [
            ("Question", query_bundle.query),
            ("Intent", query_bundle.intent),
        ]

        image_text = "\n\n".join(self._format_image_prompt_block(image) for image in images)
        citation_text = "\n\n".join(self._format_citation_prompt_block(citation) for citation in citations)

        if top_chunk_kind == "image" and image_text:
            sections.append(("Figure Evidence", image_text))

        if citation_text:
            sections.append(("Text Evidence", citation_text))

        if top_chunk_kind != "image" and image_text:
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
        candidate: RetrievalCandidate,
        *,
        card_id: str,
        citation_id_map: dict[str, str],
        is_top_ranked: bool,
    ) -> dict[str, Any] | None:
        if not candidate.image_url:
            return None
        surrounding_text = str(candidate.text or "").strip()
        caption = str(candidate.caption or "").strip()
        citation_ids = [citation_id_map[candidate_id] for candidate_id in candidate.linked_ids if candidate_id in citation_id_map]
        return {
            "id": card_id,
            "image_id": candidate.candidate_id,
            "page_number": candidate.page_numbers[0] if candidate.page_numbers else None,
            "caption": caption,
            "surrounding_text": surrounding_text,
            "primary_text": surrounding_text or caption,
            "labels": list(candidate.labels),
            "image_url": candidate.image_url,
            "cloudinary_public_id": candidate.cloudinary_public_id,
            "source_file": candidate.source_file,
            "citation_ids": citation_ids,
            "is_top_ranked": bool(is_top_ranked),
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
        if evidence.reranked_candidates and evidence.reranked_candidates[0].kind == "image":
            top_image = evidence.reranked_candidates[0]
            has_image_context = bool(str(top_image.text or "").strip() or str(top_image.caption or "").strip())
            return bool(images) and has_image_context

        if citations:
            top_score = float(evidence.citation_candidates[0].score)
            if top_score >= 0.1:
                return True
        if query_bundle.is_visual:
            top_image_score = float(evidence.image_candidates[0].score) if evidence.image_candidates else -999.0
            return bool(images) and (bool(citations) or top_image_score >= 0.35)
        if query_bundle.is_shloka:
            return any(self._is_shloka_like_candidate(candidate) for candidate in evidence.citation_candidates) and bool(citations)
        if query_bundle.is_table:
            top_table_score = float(evidence.table_candidates[0].score) if evidence.table_candidates else -999.0
            return bool(tables) and (bool(citations) or top_table_score >= 0.2)
        return False

    @staticmethod
    def _format_citation_prompt_block(citation: dict[str, Any]) -> str:
        section = " > ".join(citation.get("section_path") or [])
        section_line = f"Section: {section}\n" if section else ""
        source_label = ContextBuilder._display_source_title(citation.get("source_file"))
        page_numbers = citation.get("page_numbers") or []
        page_label = f"p.{page_numbers[0]}" if len(page_numbers) == 1 else f"pp.{', '.join(str(page) for page in page_numbers)}"
        return (
            f"[{citation['id']}] Source: {source_label}, {page_label}\n"
            f"{section_line}"
            f"{citation['snippet']}"
        ).strip()

    @staticmethod
    def _format_image_prompt_block(image: dict[str, Any]) -> str:
        labels = ", ".join(image.get("labels") or [])
        labels_line = f"\nLabels: {labels}" if labels else ""
        citation_line = f"\nLinked citations: {', '.join(image.get('citation_ids') or [])}" if image.get("citation_ids") else ""
        diagram_token = ContextBuilder._diagram_token(image)
        surrounding_text = str(image.get("surrounding_text") or "").strip()
        caption = str(image.get("caption") or "").strip()
        primary_text = surrounding_text or caption
        caption_line = f"\nCaption: {caption}" if caption else ""
        return (
            f"[{image['id']}] Source: {image['source_file']} page {image['page_number']}\n"
            f"Primary context: {primary_text}{caption_line}\n"
            f"Diagram token: {diagram_token}{labels_line}{citation_line}"
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
        if query_bundle.is_shloka and self._is_shloka_like_candidate(candidate):
            return candidate.score >= -0.15
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
            return bool(overlap) or bool(candidate.linked_ids) or candidate.score >= 0.92
        return bool(overlap) and candidate.score >= 0.2

    def _keyword_overlap(self, query_keywords: set[str], candidate: RetrievalCandidate) -> set[str]:
        if not query_keywords:
            return set()
        candidate_text = " ".join(
            part
            for part in [
                candidate.heading_text or "",
                " ".join(candidate.section_path or []),
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
        keywords: set[str] = set()
        for token in re.findall(r"\w+", str(text or "").lower()):
            if len(token) < 3 or token in self.STOPWORDS or token.isdigit():
                continue
            keywords.add(token)
            if token.endswith("s") and len(token) >= 5:
                keywords.add(token[:-1])
            if token.endswith("es") and len(token) >= 6:
                keywords.add(token[:-2])
        return keywords

    @staticmethod
    def _is_shloka_like_candidate(candidate: RetrievalCandidate) -> bool:
        payload = dict(candidate.payload or {})
        text = str(candidate.text or candidate.snippet or "").strip()
        scripts = {str(script) for script in payload.get("scripts", [])}

        if candidate.chunk_type == "shloka":
            return True
        if bool(payload.get("is_shloka")) or payload.get("shloka_number"):
            return True
        if "Deva" in scripts and any(mark in text for mark in ("।", "॥", "|")):
            return True
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return len(lines) >= 2 and all(len(line.split()) <= 16 for line in lines)

    @staticmethod
    def _display_source_title(source_file: object) -> str:
        raw = str(source_file or "").strip()
        if raw.lower().endswith(".pdf"):
            return raw[:-4]
        return raw or "Unknown Source"

    @staticmethod
    def _diagram_token(image: dict[str, Any]) -> str:
        public_id = str(image.get("cloudinary_public_id") or "").strip()
        image_id = str(image.get("image_id") or "").strip()
        source_file = str(image.get("source_file") or "").strip()
        page_number = image.get("page_number")
        base = public_id.rsplit("/", 1)[-1] if public_id else image_id or source_file or "diagram"
        if "." not in base:
            base = f"{base}.png"
        if page_number and "page" not in base.lower():
            stem, dot, suffix = base.rpartition(".")
            if dot:
                base = f"{stem}_page{page_number}.{suffix}"
        return f"[Diagram: {base}]"
