"use client";

import Link from "next/link";
import { FormEvent, startTransition, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { apiBaseUrl, QueryResponse, streamQuery, TableCard } from "./lib/query-api";

const SAMPLE_QUERIES = [
  "What is Palika Yantra?",
  "Explain the definition of rasa shastra",
  "Show me the figure for thin-layer chromatography",
];

export default function Home() {
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [statusMessage, setStatusMessage] = useState("Ready to search the indexed corpus.");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  const answer = useMemo(() => streamedAnswer || result?.answer || "", [result?.answer, streamedAnswer]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      setError("Enter a query first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setStreamedAnswer("");
    setStatusMessage("Connecting to the backend...");

    try {
      await streamQuery(
        { query: trimmed },
        {
          onStatus: (status) => {
            setStatusMessage(status.message);
          },
          onToken: (token) => {
            setStatusMessage("Streaming answer...");
            setStreamedAnswer((current) => current + token);
          },
          onFinal: (response) => {
            startTransition(() => {
              setResult(response);
              setStreamedAnswer(response.answer || "");
              setStatusMessage(
                response.enough_evidence
                  ? "Grounded answer ready."
                  : "No strong evidence found in the indexed material.",
              );
            });
          },
          onError: (message) => {
            throw new Error(message);
          },
        },
      );
    } catch (err) {
      setResult(null);
      setStreamedAnswer("");
      setStatusMessage("The request did not complete.");
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="landing-shell">
      <section className="landing-hero">
        <div className="hero-topline">
          <p className="eyebrow">Ayurvedic Multimodal RAG</p>
          <Link href="/developer" className="ghost-link">
            Open developer view
          </Link>
        </div>

        <div className="hero-copy-block">
          <h1>Ask Ayurveda-grade questions and get grounded answers, not guesswork.</h1>
          <p className="hero-copy">
            The landing page stays quiet and focused: answer first, then only the images or tables
            that actually help. The full evidence trace lives separately in the developer route.
          </p>
        </div>

        <form className="landing-form" onSubmit={handleSubmit}>
          <label className="query-label" htmlFor="query">
            Ask the system
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask about a formulation, instrument, process, figure, table, or shloka..."
            rows={4}
          />

          <div className="landing-actions">
            <button type="submit" disabled={loading}>
              {loading ? "Streaming..." : "Ask"}
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
          <span>{statusMessage}</span>
          <code>{apiBaseUrl()}</code>
        </div>
      </section>

      <section className="chat-layout">
        <article className="primary-card answer-surface">
          <div className="card-header">
            <p className="card-kicker">Answer</p>
            {result ? (
              <span className={`pill ${result.enough_evidence ? "pill-good" : "pill-warn"}`}>
                {result.enough_evidence ? "Grounded" : "Low evidence"}
              </span>
            ) : null}
          </div>

          {error ? <p className="error-text">{error}</p> : null}

          {answer ? (
            <div className="markdown-answer">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
            </div>
          ) : (
            <p className="placeholder-text">
              Ask a question to start a grounded search across the indexed Ayurveda corpus.
            </p>
          )}

          {result ? (
            <div className="meta-strip">
              <span>Intent: {result.query_intent}</span>
              <span>Model: {result.model}</span>
              <span>Total: {formatSeconds(totalTime(result.timings))}</span>
            </div>
          ) : null}
        </article>

        <aside className="support-rail">
          {result?.enough_evidence && result.tables.length ? (
            <section className="primary-card media-panel">
              <div className="card-header">
                <p className="card-kicker">Relevant Tables</p>
                <span className="pill">{result.tables.length}</span>
              </div>
              <div className="stack">
                {result.tables.map((table) => (
                  <TablePreview key={table.id} table={table} />
                ))}
              </div>
            </section>
          ) : null}

          {result?.enough_evidence && result.images.length ? (
            <section className="primary-card media-panel">
              <div className="card-header">
                <p className="card-kicker">Relevant Images</p>
                <span className="pill">{result.images.length}</span>
              </div>
              <div className="stack">
                {result.images.map((image) => (
                  <figure key={image.id} className="image-card">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={image.image_url} alt={image.caption || image.image_id} />
                    <figcaption className="image-meta">
                      <strong>{image.caption || image.id}</strong>
                      <span>
                        {image.source_file}
                        {image.page_number ? ` · page ${image.page_number}` : ""}
                      </span>
                    </figcaption>
                  </figure>
                ))}
              </div>
            </section>
          ) : null}
        </aside>
      </section>
    </main>
  );
}

function TablePreview({ table }: { table: TableCard }) {
  return (
    <article className="table-card">
      <div className="table-card-head">
        <strong>{table.table_caption || table.id}</strong>
        <span>
          {table.source_file}
          {table.page_numbers.length ? ` · p${table.page_numbers.join(", ")}` : ""}
        </span>
      </div>

      {table.table_markdown ? (
        <div className="markdown-table">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{table.table_markdown}</ReactMarkdown>
        </div>
      ) : null}
    </article>
  );
}

function formatSeconds(value: number | undefined): string {
  return `${Number(value || 0).toFixed(2)}s`;
}

function totalTime(timings: Record<string, number>): number {
  return Number(timings.total_pre_llm_sec || 0) + Number(timings.llm_sec || 0);
}
