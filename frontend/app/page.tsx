"use client";

import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Citation, ImageCard, QueryResponse, TableCard, streamQuery } from "./lib/query-api";

const MODEL_NAME = "llama-3.3-70b-versatile";
const STARTER_CONVERSATIONS = [
  "What is Nasya therapy?",
  "Explain Agni in digestion",
  "Show me the figure for Palika Yantra",
];

const SCROLL_PADDING = 24;

type ChatRole = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  status?: string;
  streaming?: boolean;
  metadata?: QueryResponse | null;
  error?: boolean;
};

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [footerHeight, setFooterHeight] = useState(144);
  const [selectedImage, setSelectedImage] = useState<ImageCard | null>(null);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const footerRef = useRef<HTMLElement | null>(null);
  const shouldAutoScrollRef = useRef(true);
  const touchStartYRef = useRef<number | null>(null);

  const hasMessages = messages.length > 0;
  const canSend = Boolean(query.trim()) && !isSubmitting;
  const isStreaming = messages.some((message) => message.streaming);

  useEffect(() => {
    if (!selectedImage) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelectedImage(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedImage]);

  useEffect(() => {
    const footerNode = footerRef.current;
    if (!footerNode) {
      return;
    }

    const updateFooterHeight = () => {
      setFooterHeight(footerNode.offsetHeight || 144);
    };

    updateFooterHeight();

    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateFooterHeight);
      return () => window.removeEventListener("resize", updateFooterHeight);
    }

    const observer = new ResizeObserver(() => updateFooterHeight());
    observer.observe(footerNode);
    window.addEventListener("resize", updateFooterHeight);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateFooterHeight);
    };
  }, []);

  useEffect(() => {
    const node = chatScrollRef.current;
    if (!node) {
      return;
    }

    const stopAutoScroll = () => {
      shouldAutoScrollRef.current = false;
    };

    const handleScroll = () => {
      const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
      if (distanceFromBottom <= 80) {
        shouldAutoScrollRef.current = true;
        return;
      }
      if (distanceFromBottom >= 140) {
        shouldAutoScrollRef.current = false;
      }
    };

    const handleWheel = (event: WheelEvent) => {
      if (event.deltaY < 0) {
        stopAutoScroll();
      }
    };

    const handleTouchStart = (event: TouchEvent) => {
      touchStartYRef.current = event.touches[0]?.clientY ?? null;
    };

    const handleTouchMove = (event: TouchEvent) => {
      const currentY = event.touches[0]?.clientY;
      if (currentY == null) {
        return;
      }
      const previousY = touchStartYRef.current;
      if (previousY != null && currentY > previousY + 4) {
        stopAutoScroll();
      }
      touchStartYRef.current = currentY;
    };

    const handleTouchEnd = () => {
      touchStartYRef.current = null;
    };

    handleScroll();
    node.addEventListener("scroll", handleScroll, { passive: true });
    node.addEventListener("wheel", handleWheel, { passive: true });
    node.addEventListener("touchstart", handleTouchStart, { passive: true });
    node.addEventListener("touchmove", handleTouchMove, { passive: true });
    node.addEventListener("touchend", handleTouchEnd, { passive: true });

    return () => {
      node.removeEventListener("scroll", handleScroll);
      node.removeEventListener("wheel", handleWheel);
      node.removeEventListener("touchstart", handleTouchStart);
      node.removeEventListener("touchmove", handleTouchMove);
      node.removeEventListener("touchend", handleTouchEnd);
    };
  }, []);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      if (!shouldAutoScrollRef.current) {
        return;
      }
      if (chatScrollRef.current) {
        chatScrollRef.current.scrollTo({
          top: chatScrollRef.current.scrollHeight + SCROLL_PADDING,
          behavior: isStreaming ? "auto" : "smooth",
        });
        return;
      }
      bottomRef.current?.scrollIntoView({ behavior: isStreaming ? "auto" : "smooth", block: "end" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [isStreaming, messages]);

  async function handleSend(nextQuery?: string) {
    const trimmed = (nextQuery ?? query).trim();
    if (!trimmed || isSubmitting) {
      return;
    }

    console.log("Chat query:", trimmed);

    const userMessageId = createMessageId("user");
    const assistantMessageId = createMessageId("assistant");

    setMessages((current) => [
      ...current,
      { id: userMessageId, role: "user", content: trimmed },
      {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        status: "Searching the indexed corpus...",
        streaming: true,
        metadata: null,
      },
    ]);
    setQuery("");
    shouldAutoScrollRef.current = true;
    setIsSubmitting(true);

    try {
      await streamQuery(
        { query: trimmed },
        {
          onStatus: (status) => {
            patchMessage(assistantMessageId, { status: status.message });
          },
          onToken: (token) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantMessageId
                  ? {
                      ...message,
                      content: `${message.content}${token}`,
                      status: "Streaming answer...",
                      streaming: true,
                    }
                  : message,
              ),
            );
          },
          onFinal: (response) => {
            patchMessage(assistantMessageId, {
              content: response.answer,
              status: response.enough_evidence ? undefined : "Insufficient evidence",
              streaming: false,
              metadata: response,
            });
          },
          onError: (message) => {
            throw new Error(message);
          },
        },
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      patchMessage(assistantMessageId, {
        content: message,
        status: "Request failed",
        streaming: false,
        error: true,
        metadata: null,
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  function patchMessage(messageId: string, patch: Partial<ChatMessage>) {
    setMessages((current) =>
      current.map((message) => (message.id === messageId ? { ...message, ...patch } : message)),
    );
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void handleSend();
  }

  function handleKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  }

  const title = useMemo(() => (hasMessages ? "Ayurvedic AI" : "How can I help you today?"), [hasMessages]);

  return (
    <main className="h-dvh overflow-hidden overscroll-none bg-[#040704] text-[#f5f3eb]">
      <div className="pointer-events-none fixed inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(34,197,94,0.18),transparent_22%),radial-gradient(circle_at_bottom_left,rgba(22,163,74,0.16),transparent_24%),linear-gradient(140deg,#020402_0%,#07110a_48%,#0b2d12_100%)]" />
        <div className="absolute inset-0 opacity-30 [background-image:linear-gradient(rgba(74,222,128,0.07)_1px,transparent_1px),linear-gradient(90deg,rgba(74,222,128,0.07)_1px,transparent_1px)] [background-size:72px_72px]" />
      </div>

      <div className="relative flex h-dvh min-h-0 flex-col overflow-hidden">
        <header className="sticky top-0 z-20 shrink-0 border-b border-white/6 bg-[#050805]/75 backdrop-blur-xl">
          <div className="mx-auto flex w-full max-w-4xl items-center justify-between px-4 py-4 sm:px-6">
            <div
              className="whitespace-nowrap text-[1.35rem] font-semibold leading-none tracking-[0.14em] text-[#f6eed9] [text-shadow:0_0_22px_rgba(207,172,87,0.16)] sm:text-[1.55rem]"
              style={{ fontFamily: 'Georgia, "Times New Roman", serif' }}
            >
              <span>Ayur</span>
              <span className="ml-2 text-emerald-300">AI</span>
            </div>
            <div className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-emerald-300">
              {MODEL_NAME}
            </div>
          </div>
        </header>

        <section
          ref={chatScrollRef}
          className="chat-scroll-region min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-y-contain"
        >
          <div
            className="mx-auto flex min-h-full w-full max-w-4xl flex-col px-4 pt-6 sm:px-6"
            style={{ paddingBottom: `calc(${footerHeight + 32}px + env(safe-area-inset-bottom))` }}
          >
            {!hasMessages ? (
              <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col items-center justify-center py-6 text-center">
              <h1 className="max-w-3xl text-balance text-4xl font-medium tracking-[-0.07em] text-white sm:text-6xl">
                {title}
              </h1>
              <p className="mt-4 max-w-xl text-sm leading-7 text-white/55 sm:text-base">
                Ask in English, Hindi, or Sanskrit and the answer will stream in with grounded evidence.
              </p>

              <div className="mt-8 flex w-full max-w-3xl flex-wrap items-center justify-center gap-3">
                {STARTER_CONVERSATIONS.map((sample) => (
                  <button
                    key={sample}
                    type="button"
                    onClick={() => setQuery(sample)}
                    className="rounded-full border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/80 transition hover:border-emerald-400/30 hover:bg-emerald-400/12 hover:text-white"
                  >
                    {sample}
                  </button>
                ))}
              </div>
              </div>
            ) : (
              <div className="flex min-h-full flex-col py-2">
                <div className="mt-auto flex flex-col gap-4">
                  {messages.map((message) => (
                    <ChatBubble key={message.id} message={message} onImageClick={setSelectedImage} />
                  ))}
                  <div ref={bottomRef} />
                </div>
              </div>
            )}
          </div>
        </section>

        <footer ref={footerRef} className="fixed inset-x-0 bottom-0 z-30 bg-gradient-to-t from-[#040704] via-[#040704]/96 to-transparent pt-4">
          <div className="mx-auto flex w-full max-w-4xl flex-col gap-2 px-4 pb-4 sm:px-6 sm:pb-5">
            <form
              onSubmit={handleSubmit}
              className="mx-auto flex w-full items-end gap-3 rounded-[1.8rem] bg-[#111611]/92 px-3 py-3 shadow-[0_18px_55px_rgba(0,0,0,0.34)] ring-1 ring-white/8 backdrop-blur-xl"
            >
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                placeholder="Ask anything..."
                className="max-h-44 min-h-[60px] flex-1 resize-none border-0 bg-transparent px-3 py-3 text-base text-white outline-none placeholder:text-white/34"
              />

              <button
                type="submit"
                disabled={!canSend}
                className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-emerald-400 text-[#06210d] transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-white/30"
                aria-label="Send message"
              >
                <span className="text-xl">↑</span>
              </button>
            </form>
          </div>
        </footer>

        {selectedImage ? <ImageLightbox image={selectedImage} onClose={() => setSelectedImage(null)} /> : null}
      </div>
    </main>
  );
}

function ChatBubble({
  message,
  onImageClick,
}: {
  message: ChatMessage;
  onImageClick: (image: ImageCard) => void;
}) {
  const isUser = message.role === "user";
  const metadata = message.metadata;
  const citations = metadata?.citations || [];
  const images = metadata?.images || [];
  const tables = metadata?.tables || [];

  return (
    <article className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-[1.7rem] px-5 py-4 shadow-[0_18px_48px_rgba(0,0,0,0.22)] sm:max-w-[78%] ${
          isUser
            ? "rounded-br-md border border-emerald-300/12 bg-emerald-400/12 text-white"
            : "rounded-bl-md border border-white/8 bg-[#121712]/94 text-white/92"
        }`}
      >
        {message.content ? (
          <div
            className={`prose max-w-none ${isUser ? "prose-invert prose-p:my-0 prose-p:leading-8" : "prose-invert prose-p:leading-8 prose-li:leading-8 prose-strong:text-white prose-code:text-emerald-200"}`}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        ) : null}

        {!message.content && message.streaming ? (
          <div className="flex items-center gap-3 text-sm text-white/46">
            <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-emerald-400" />
            {message.status || "Waiting for response..."}
          </div>
        ) : null}

        {!isUser && message.status && !message.streaming ? (
          <p className={`mt-3 text-[11px] uppercase tracking-[0.2em] ${message.error ? "text-rose-300/80" : "text-white/34"}`}>
            {message.status}
          </p>
        ) : null}

        {!isUser && (images.length || tables.length || citations.length) ? (
          <div className="mt-5 space-y-4 border-t border-white/6 pt-4">
            {images.length ? <ImageStrip images={images} onImageClick={onImageClick} /> : null}
            {tables.length ? <TableStrip tables={tables} /> : null}
            {citations.length ? <CitationStrip citations={citations} /> : null}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function ImageStrip({
  images,
  onImageClick,
}: {
  images: ImageCard[];
  onImageClick: (image: ImageCard) => void;
}) {
  return (
    <div className="space-y-2">
      <p className="text-[11px] uppercase tracking-[0.22em] text-emerald-300/72">Images</p>
      <div className="grid gap-3 sm:grid-cols-2">
        {images.map((image) => (
          <button
            key={image.id}
            type="button"
            onClick={() => onImageClick(image)}
            className="overflow-hidden rounded-2xl border border-white/8 bg-black/18 text-left transition hover:border-emerald-300/35 hover:bg-black/28"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={image.image_url} alt={image.caption || image.image_id} className="h-40 w-full object-cover" />
            <div className="space-y-1 px-3 py-3 text-sm">
              <p className="text-white/90">{image.caption || image.id}</p>
              <p className="text-white/45">
                {image.source_file}
                {image.page_number ? ` · p.${image.page_number}` : ""}
              </p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function ImageLightbox({ image, onClose }: { image: ImageCard; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/82 p-4 backdrop-blur-md sm:p-8" onClick={onClose}>
      <div
        className="relative flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-[2rem] border border-white/10 bg-[#09110b]/96 shadow-[0_32px_80px_rgba(0,0,0,0.55)]"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-4 border-b border-white/8 px-5 py-4">
          <div className="min-w-0">
            <p className="truncate text-base font-medium text-white">{image.caption || image.image_id}</p>
            <p className="truncate text-sm text-white/50">
              {image.source_file}
              {image.page_number ? ` · p.${image.page_number}` : ""}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full border border-white/12 bg-white/6 px-4 py-2 text-sm text-white/80 transition hover:bg-white/10"
          >
            Close
          </button>
        </div>

        <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto bg-[radial-gradient(circle_at_top,rgba(16,185,129,0.08),transparent_42%),linear-gradient(180deg,rgba(7,16,10,0.98),rgba(2,7,4,0.98))] p-4 sm:p-6">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={image.image_url}
            alt={image.caption || image.image_id}
            className="h-auto max-h-full w-auto max-w-full rounded-2xl object-contain shadow-[0_18px_50px_rgba(0,0,0,0.35)]"
          />
        </div>
      </div>
    </div>
  );
}

function TableStrip({ tables }: { tables: TableCard[] }) {
  return (
    <div className="space-y-2">
      <p className="text-[11px] uppercase tracking-[0.22em] text-emerald-300/72">Tables</p>
      <div className="space-y-3">
        {tables.map((table) => (
          <div key={table.id} className="overflow-hidden rounded-2xl border border-white/8 bg-black/18">
            <div className="border-b border-white/6 px-3 py-2 text-sm text-white/72">
              {table.table_caption || table.id} · {table.source_file}
            </div>
            <div className="overflow-x-auto px-3 py-3">
              <div className="prose prose-invert max-w-none prose-table:text-sm prose-th:border prose-td:border prose-th:border-white/10 prose-td:border-white/10">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{table.table_markdown || ""}</ReactMarkdown>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CitationStrip({ citations }: { citations: Citation[] }) {
  return (
    <div className="space-y-2">
      <p className="text-[11px] uppercase tracking-[0.22em] text-emerald-300/72">Citations</p>
      <div className="flex flex-wrap gap-2">
        {citations.map((citation) => (
          <div key={citation.id} className="rounded-full border border-white/8 bg-white/4 px-3 py-2 text-xs text-white/72">
            <span className="font-semibold text-white/88">{citation.id}</span>
            <span className="ml-2">
              {displayBookTitle(citation.source_file)}
              {citation.page_numbers.length ? ` · p.${citation.page_numbers.join(", ")}` : ""}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function displayBookTitle(sourceFile: string): string {
  return sourceFile.replace(/\.pdf$/i, "");
}

function createMessageId(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}
