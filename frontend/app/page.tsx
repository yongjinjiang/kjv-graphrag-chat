"use client";

import { useEffect, useRef, useState } from "react";

import { GraphPanel, type Subgraph } from "./components/GraphPanel";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Citation = {
  ref: string;
  text: string;
  verse_id: string;
};

type QueryResponse = {
  question: string;
  mode: string;
  answer: string;
  citations: Citation[];
  graph_context: string[];
  subgraph: Subgraph;
};

type ChatMessage =
  | { role: "user"; content: string }
  | {
      role: "assistant";
      content: string;
      mode: string;
      citations: Citation[];
      graph_context: string[];
      subgraph: Subgraph | null;
    };

type RightTab = "graph" | "sources";

const SUGGESTIONS = [
  "Genesis 1:1",
  "Who was Moses?",
  "What does the Bible say about forgiveness?",
  "How is David related to Goliath?",
  "Tell me about the crossing of the Red Sea",
];

const MODE_COLORS: Record<string, string> = {
  lookup: "bg-blue-100 text-blue-800",
  entity: "bg-red-100 text-red-800",
  theme: "bg-purple-100 text-purple-800",
  relation: "bg-orange-100 text-orange-800",
  semantic: "bg-gray-100 text-gray-800",
  hybrid: "bg-green-100 text-green-800",
};

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<RightTab>("graph");
  const scrollRef = useRef<HTMLDivElement>(null);

  const lastAssistant = [...messages]
    .reverse()
    .find(
      (m): m is Extract<ChatMessage, { role: "assistant" }> => m.role === "assistant",
    );

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, loading]);

  async function send(question: string) {
    if (!question.trim() || loading) return;
    setError(null);
    setMessages((m) => [...m, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k: 25 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      const data: QueryResponse = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer,
          mode: data.mode,
          citations: data.citations,
          graph_context: data.graph_context,
          subgraph: data.subgraph ?? null,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex-1 flex flex-col bg-gray-50">
      <header className="border-b border-gray-200 bg-white px-6 py-3">
        <h1 className="text-lg font-semibold text-gray-900">
          BibleGraphRAG <span className="text-gray-400 font-normal">· KJV</span>
        </h1>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-0 min-h-0">
        {/* Chat column */}
        <section className="lg:col-span-2 flex flex-col min-h-0 border-r border-gray-200 bg-white">
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
            {messages.length === 0 && (
              <div className="text-center text-gray-500 mt-12">
                <p className="mb-4">Ask a question about the King James Bible.</p>
                <div className="flex flex-wrap gap-2 justify-center max-w-2xl mx-auto">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => send(s)}
                      className="text-sm px-3 py-1.5 rounded-full border border-gray-300 hover:bg-gray-100 text-gray-700"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div
                key={i}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-2.5 whitespace-pre-wrap text-[15px] leading-relaxed ${
                    m.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-900"
                  }`}
                >
                  {m.role === "assistant" && (
                    <div className="mb-1.5 flex items-center gap-2 text-xs">
                      <span
                        className={`px-2 py-0.5 rounded-full font-medium ${
                          MODE_COLORS[m.mode] ?? "bg-gray-200 text-gray-700"
                        }`}
                      >
                        {m.mode}
                      </span>
                      <span className="text-gray-500">
                        {m.citations.length} citation
                        {m.citations.length === 1 ? "" : "s"}
                      </span>
                    </div>
                  )}
                  {m.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-500 rounded-2xl px-4 py-2.5 text-sm">
                  <span className="inline-block animate-pulse">Searching the graph…</span>
                </div>
              </div>
            )}

            {error && (
              <div className="rounded-md border border-red-300 bg-red-50 text-red-800 px-3 py-2 text-sm">
                {error}
              </div>
            )}
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
            className="border-t border-gray-200 p-3 bg-white"
          >
            <div className="flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question…"
                disabled={loading}
                className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50"
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="rounded-lg bg-blue-600 text-white px-4 py-2 text-sm font-medium hover:bg-blue-700 disabled:bg-gray-300"
              >
                Send
              </button>
            </div>
          </form>
        </section>

        {/* Right column: tabbed Graph / Sources */}
        <aside className="hidden lg:flex flex-col min-h-0 bg-gray-50">
          <div className="px-3 py-2 border-b border-gray-200 bg-white flex items-center gap-1">
            {(["graph", "sources"] as const).map((tab) => {
              const active = rightTab === tab;
              return (
                <button
                  key={tab}
                  onClick={() => setRightTab(tab)}
                  className={`text-sm px-3 py-1.5 rounded-md transition-colors ${
                    active
                      ? "bg-gray-900 text-white"
                      : "text-gray-600 hover:bg-gray-100"
                  }`}
                >
                  {tab === "graph" ? "Graph" : "Sources"}
                  {tab === "sources" && lastAssistant && (
                    <span
                      className={`ml-1.5 text-[10px] px-1.5 py-0.5 rounded-full ${
                        active ? "bg-white/20" : "bg-gray-200 text-gray-700"
                      }`}
                    >
                      {lastAssistant.citations.length}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {rightTab === "graph" ? (
            <GraphPanel subgraph={lastAssistant?.subgraph} />
          ) : (
            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
              {!lastAssistant && (
                <p className="text-sm text-gray-500">
                  Citations and graph context for the most recent answer will
                  appear here.
                </p>
              )}
              {lastAssistant && (
                <>
                  {lastAssistant.graph_context.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
                        Graph context
                      </h3>
                      <ul className="space-y-1 text-sm text-gray-700">
                        {lastAssistant.graph_context.map((c, i) => (
                          <li key={i} className="leading-snug">
                            • {c}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
                      Citations ({lastAssistant.citations.length})
                    </h3>
                    <ul className="space-y-2">
                      {lastAssistant.citations.map((c) => (
                        <li
                          key={c.verse_id}
                          className="rounded-md border border-gray-200 bg-white p-3"
                        >
                          <div className="text-xs font-semibold text-blue-700 mb-0.5">
                            {c.ref}
                          </div>
                          <div className="text-sm text-gray-800 leading-snug">
                            {c.text}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                </>
              )}
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
