"use client";

export type Citation = {
  id: string;
  kind: string;
  doc_id: string;
  source_file: string;
  page_numbers: number[];
  section_path: string[];
  snippet: string;
};

export type ImageCard = {
  id: string;
  image_id: string;
  page_number?: number | null;
  caption: string;
  labels: string[];
  image_url: string;
  cloudinary_public_id?: string | null;
  source_file: string;
  citation_ids?: string[];
};

export type TableCard = {
  id: string;
  table_id: string;
  page_numbers: number[];
  table_caption?: string | null;
  table_markdown?: string | null;
  table_rows?: string[][] | null;
  source_file: string;
  citation_ids?: string[];
};

export type DebugCandidate = {
  id: string;
  kind: string;
  score: number;
  doc_id: string;
  source_file: string;
  page_numbers: number[];
  chunk_type?: string | null;
  image_type?: string | null;
  section_path: string[];
  snippet: string;
  retrieval_reasons: string[];
  linked_ids: string[];
  table_markdown?: string | null;
  caption?: string | null;
  image_url?: string | null;
};

export type RetrievalTiming = {
  embed_query_sec?: number;
  text_original_sec?: number;
  text_normalized_sec?: number;
  image_direct_sec?: number;
  linked_image_rescue_sec?: number;
  page_proximity_rescue_sec?: number;
  hydrate_points_sec?: number;
  merge_sort_sec?: number;
};

export type RetrievalCounts = {
  text_hits_original?: number;
  text_hits_normalized?: number;
  image_hits_direct?: number;
  rescued_image_hits?: number;
  rescued_page_hits?: number;
  candidate_count_merged?: number;
};

export type RerankTiming = {
  pair_build_sec?: number;
  model_infer_sec?: number;
  postprocess_sec?: number;
};

export type RerankMeta = {
  device?: string;
  model?: string;
  pool_size?: number;
  warm_model?: boolean;
};

export type QueryDebug = {
  query_bundle?: Record<string, unknown>;
  retrieved_candidates?: DebugCandidate[];
  reranked_candidates?: DebugCandidate[];
  retrieval_timing?: RetrievalTiming;
  retrieval_counts?: RetrievalCounts;
  rerank_timing?: RerankTiming;
  rerank_meta?: RerankMeta;
  retrieval?: Record<string, unknown>;
  rerank?: Record<string, unknown>;
  context?: Record<string, unknown>;
  final_context?: {
    citations: Citation[];
    images: ImageCard[];
    tables: TableCard[];
    enough_evidence: boolean;
    user_prompt?: string;
  };
};

export type QueryResponse = {
  answer: string;
  citations: Citation[];
  images: ImageCard[];
  tables: TableCard[];
  enough_evidence: boolean;
  query_intent: string;
  model: string;
  timings: Record<string, number>;
  debug?: QueryDebug | null;
};

export type QueryRequest = {
  query: string;
  include_debug?: boolean;
  doc_id?: string;
  page_start?: number;
  page_end?: number;
  languages?: string[];
  scripts?: string[];
  chunk_types?: string[];
};

type StreamStatus = {
  stage: string;
  message: string;
};

type StreamHandlers = {
  onToken?: (token: string) => void;
  onStatus?: (status: StreamStatus) => void;
  onFinal?: (response: QueryResponse) => void;
  onError?: (message: string) => void;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export function apiBaseUrl(): string {
  return API_BASE_URL;
}

export async function runQuery(payload: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Request failed");
  }

  return (await response.json()) as QueryResponse;
}

export async function streamQuery(payload: QueryRequest, handlers: StreamHandlers): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Streaming request failed");
  }

  if (!response.body) {
    throw new Error("Streaming response body is unavailable");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.replace(/\r/g, "").split("\n\n");
    buffer = frames.pop() || "";

    for (const frame of frames) {
      const parsed = parseSseFrame(frame);
      if (!parsed) {
        continue;
      }

      if (parsed.event === "token") {
        handlers.onToken?.(String(parsed.data ?? ""));
        continue;
      }

      if (parsed.event === "status") {
        handlers.onStatus?.(parsed.data as StreamStatus);
        continue;
      }

      if (parsed.event === "final") {
        handlers.onFinal?.(parsed.data as QueryResponse);
        continue;
      }

      if (parsed.event === "error") {
        const detail =
          typeof parsed.data === "string"
            ? parsed.data
            : String((parsed.data as Record<string, unknown>)?.detail || "Streaming request failed");
        handlers.onError?.(detail);
      }
    }
  }

  const trailing = parseSseFrame(buffer);
  if (trailing?.event === "final") {
    handlers.onFinal?.(trailing.data as QueryResponse);
  }
}

function parseSseFrame(frame: string): { event: string; data: unknown } | null {
  const lines = frame
    .split("\n")
    .map((line) => line.trimEnd())
    .filter(Boolean);

  if (!lines.length) {
    return null;
  }

  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  const rawData = dataLines.join("\n");
  return {
    event,
    data: parseSseData(rawData),
  };
}

function parseSseData(raw: string): unknown {
  if (!raw) {
    return "";
  }
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}
