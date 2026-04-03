"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";

import { apiBaseUrl, DebugCandidate, QueryResponse, runQuery } from "../lib/query-api";

const SAMPLE_QUERIES = [
  "What is Palika Yantra?",
  "Show me the figure for thin-layer chromatography",
  "What does the table say about gemstone hardness?",
  "cricket",
];

export default function DeveloperPage() {
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
      const response = await runQuery({ query: trimmed, include_debug: true });
      setResult(response);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const debug = result?.debug;

  return (
    <main className="developer-shell">
      <section className="developer-hero">
        <div className="hero-topline">
          <p className="eyebrow">Developer View</p>
          <Link href="/" className="ghost-link">
            Back to product view
          </Link>
        </div>

        <h1>Inspect retrieval, reranking, and the final grounded context.</h1>
        <p className="hero-copy">
          This route is for local debugging. It shows the answer, the surfaced evidence, the raw
          retrieved candidates, the reranked order, and the final prompt context sent to the LLM.
        </p>

        <form className="landing-form" onSubmit={handleSubmit}>
          <label className="query-label" htmlFor="developer-query">
            Debug query
          </label>
          <textarea
            id="developer-query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={4}
            placeholder="Run a query and inspect the entire retrieval pipeline..."
          />

          <div className="landing-actions">
            <button type="submit" disabled={loading}>
              {loading ? "Inspecting..." : "Inspect pipeline"}
            </button>

            <div className="chips">
              {SAMPLE_QUERIES.map((sample) => (
                <button key={sample} type="button" className="chip" onClick={() => setQuery(sample)}>
                  {sample}
                </button>
              ))}
            </div>
          </div>
        </form>

        <div className="status-bar">
          <span>{loading ? "Running full query pipeline..." : "Ready for inspection."}</span>
          <code>{apiBaseUrl()}</code>
        </div>
      </section>

      {error ? <p className="error-text developer-error">{error}</p> : null}

      <section className="developer-grid">
        <article className="primary-card">
          <div className="card-header">
            <p className="card-kicker">Answer</p>
            {result ? (
              <span className={`pill ${result.enough_evidence ? "pill-good" : "pill-warn"}`}>
                {result.enough_evidence ? "Grounded" : "Low evidence"}
              </span>
            ) : null}
          </div>

          {result ? (
            <>
              <p className="answer-text">{result.answer}</p>
              <div className="meta-grid">
                <span>Intent: {result.query_intent}</span>
                <span>Model: {result.model}</span>
                <span>Pre-LLM: {formatSeconds(result.timings.total_pre_llm_sec)}</span>
                <span>LLM: {formatSeconds(result.timings.llm_sec)}</span>
              </div>
            </>
          ) : (
            <p className="placeholder-text">Run a query to populate the debug view.</p>
          )}
        </article>

        <article className="primary-card">
          <div className="card-header">
            <p className="card-kicker">Returned Evidence</p>
            <span className="pill">
              {result ? `${result.citations.length}C / ${result.images.length}F / ${result.tables.length}T` : "0"}
            </span>
          </div>

          <div className="stack">
            {result?.citations.map((citation) => (
              <div key={citation.id} className="citation-item">
                <div className="citation-head">
                  <strong>{citation.id}</strong>
                  <span>
                    {citation.source_file}
                    {citation.page_numbers.length ? ` · p${citation.page_numbers.join(", ")}` : ""}
                  </span>
                </div>
                <p>{citation.snippet}</p>
              </div>
            ))}

            {result?.images.map((image) => (
              <figure key={image.id} className="image-card">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={image.image_url} alt={image.caption || image.image_id} />
                <figcaption className="image-meta">
                  <strong>{image.id}</strong>
                  <p>{image.caption || "No caption available."}</p>
                  <span>
                    {image.source_file}
                    {image.page_number ? ` · page ${image.page_number}` : ""}
                  </span>
                </figcaption>
              </figure>
            ))}

            {result?.tables.map((table) => (
              <div key={table.id} className="table-card">
                <div className="table-card-head">
                  <strong>{table.table_caption || table.id}</strong>
                  <span>{table.source_file}</span>
                </div>
                <pre className="debug-pre">{table.table_markdown || "(No markdown rendered)"}</pre>
              </div>
            ))}

            {!result?.citations.length && !result?.images.length && !result?.tables.length ? (
              <p className="placeholder-text">No evidence objects were returned in the public response.</p>
            ) : null}
          </div>
        </article>
      </section>

      <section className="developer-grid developer-grid-deep">
        <DebugColumn title="Retrieved candidates" items={debug?.retrieved_candidates || []} />
        <DebugColumn title="Reranked candidates" items={debug?.reranked_candidates || []} />
      </section>

      <section className="developer-grid developer-grid-deep">
        <article className="primary-card">
          <div className="card-header">
            <p className="card-kicker">Final Context</p>
            <span className="pill">{debug?.final_context?.enough_evidence ? "Usable" : "Thin"}</span>
          </div>
          <pre className="debug-pre">{debug?.final_context?.user_prompt || "No final prompt captured yet."}</pre>
        </article>

        <article className="primary-card">
          <div className="card-header">
            <p className="card-kicker">Debug Stats</p>
          </div>
          <pre className="debug-pre">
            {JSON.stringify(
              {
                query_bundle: debug?.query_bundle || null,
                retrieval: debug?.retrieval || null,
                rerank: debug?.rerank || null,
                context: debug?.context || null,
              },
              null,
              2,
            )}
          </pre>
        </article>
      </section>
    </main>
  );
}

function DebugColumn({ title, items }: { title: string; items: DebugCandidate[] }) {
  return (
    <article className="primary-card">
      <div className="card-header">
        <p className="card-kicker">{title}</p>
        <span className="pill">{items.length}</span>
      </div>

      <div className="stack">
        {items.length ? (
          items.map((item) => (
            <div key={item.id} className="debug-candidate">
              <div className="citation-head">
                <strong>{item.id}</strong>
                <span>{item.kind}</span>
              </div>
              <p className="debug-score">score {item.score.toFixed(3)}</p>
              <p>{item.snippet || "(No snippet available)"}</p>
              <p className="debug-meta">
                {item.source_file}
                {item.page_numbers.length ? ` · p${item.page_numbers.join(", ")}` : ""}
              </p>
              {item.retrieval_reasons.length ? (
                <p className="debug-meta">reasons: {item.retrieval_reasons.join(", ")}</p>
              ) : null}
            </div>
          ))
        ) : (
          <p className="placeholder-text">No candidates available.</p>
        )}
      </div>
    </article>
  );
}

function formatSeconds(value: number | undefined): string {
  return `${Number(value || 0).toFixed(2)}s`;
}
