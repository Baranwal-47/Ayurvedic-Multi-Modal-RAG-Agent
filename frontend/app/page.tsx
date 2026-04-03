"use client";

import { FormEvent, useState } from "react";

type Citation = {
  id: string;
  kind: string;
  source_file: string;
  page_numbers: number[];
  section_path: string[];
  snippet: string;
};

type ImageCard = {
  id: string;
  image_id: string;
  page_number?: number | null;
  caption: string;
  labels: string[];
  image_url: string;
  source_file: string;
};

type QueryResponse = {
  answer: string;
  citations: Citation[];
  images: ImageCard[];
  tables: unknown[];
  enough_evidence: boolean;
  query_intent: string;
  model: string;
  timings: Record<string, number>;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

const SAMPLE_QUERIES = [
  "What is Palika Yantra?",
  "Show me the figure for thin-layer chromatography",
  "Give me the definition of rasa shastra",
];

export default function Home() {
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      setError("Enter a query first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: trimmed }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Request failed");
      }

      const data = (await response.json()) as QueryResponse;
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <p className="eyebrow">Ayurveda Multimodal RAG</p>
        <h1>Ask the backend and inspect the evidence it returns.</h1>
        <p className="hero-copy">
          This local frontend sends a simple <code>POST /query</code> request and renders the
          answer, citations, and figure cards from the backend.
        </p>

        <form className="query-form" onSubmit={handleSubmit}>
          <label className="query-label" htmlFor="query">
            Query
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask about a formulation, shloka, apparatus, figure, or procedure..."
            rows={5}
          />

          <div className="query-actions">
            <button type="submit" disabled={loading}>
              {loading ? "Querying..." : "Send /query"}
            </button>

            <div className="chips">
              {SAMPLE_QUERIES.map((sample) => (
                <button
                  key={sample}
                  type="button"
                  className="chip"
                  onClick={() => setQuery(sample)}
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        </form>

        <div className="status-row">
          <span>API</span>
          <code>{API_BASE_URL}</code>
        </div>
      </section>

      <section className="results-grid">
        <article className="result-card answer-card">
          <div className="card-header">
            <p className="card-kicker">Answer</p>
            {result ? (
              <span className={`pill ${result.enough_evidence ? "pill-good" : "pill-warn"}`}>
                {result.enough_evidence ? "Grounded" : "Low evidence"}
              </span>
            ) : null}
          </div>

          {error ? <p className="error-text">{error}</p> : null}

          {result ? (
            <>
              <p className="answer-text">{result.answer}</p>
              <div className="meta-row">
                <span>Intent: {result.query_intent}</span>
                <span>Model: {result.model}</span>
              </div>
            </>
          ) : (
            <p className="placeholder-text">
              Submit a query to render the answer and supporting evidence here.
            </p>
          )}
        </article>

        <article className="result-card">
          <div className="card-header">
            <p className="card-kicker">Citations</p>
            <span className="pill">{result?.citations.length ?? 0}</span>
          </div>

          <div className="stack">
            {result?.citations.length ? (
              result.citations.map((citation) => (
                <div key={citation.id} className="citation-item">
                  <div className="citation-head">
                    <strong>{citation.id}</strong>
                    <span>
                      {citation.source_file} · p{citation.page_numbers.join(", ")}
                    </span>
                  </div>
                  {citation.section_path.length ? (
                    <p className="section-path">{citation.section_path.join(" > ")}</p>
                  ) : null}
                  <p>{citation.snippet}</p>
                </div>
              ))
            ) : (
              <p className="placeholder-text">No citations yet.</p>
            )}
          </div>
        </article>

        <article className="result-card">
          <div className="card-header">
            <p className="card-kicker">Images</p>
            <span className="pill">{result?.images.length ?? 0}</span>
          </div>

          <div className="stack">
            {result?.images.length ? (
              result.images.map((image) => (
                <div key={image.id} className="image-card">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={image.image_url} alt={image.caption || image.image_id} />
                  <div className="image-meta">
                    <strong>{image.id}</strong>
                    <p>{image.caption || "No caption available."}</p>
                    <span>
                      {image.source_file}
                      {image.page_number ? ` · page ${image.page_number}` : ""}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <p className="placeholder-text">No image cards returned for this query.</p>
            )}
          </div>
        </article>
      </section>
    </main>
  );
}
