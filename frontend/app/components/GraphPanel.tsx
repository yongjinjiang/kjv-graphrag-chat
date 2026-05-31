"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import cytoscape, { Core, ElementDefinition } from "cytoscape";
// @ts-expect-error — cytoscape-fcose has no bundled types
import fcose from "cytoscape-fcose";

if (typeof window !== "undefined") {
  // Register the layout once per page load; cytoscape.use is idempotent.
  try {
    cytoscape.use(fcose);
  } catch {
    /* already registered */
  }
}

export type SubgraphNode = {
  id: string;
  kind?: string;
  label?: string;
  name?: string;
  ref?: string;
  text?: string;
  book?: string;
  chapter?: number;
  verse?: number;
};

export type SubgraphLink = {
  source: string;
  target: string;
  kind?: string;
};

export type Subgraph = {
  directed?: boolean;
  multigraph?: boolean;
  nodes: SubgraphNode[];
  links: SubgraphLink[];
};

// Slightly punchier palette than the pyvis original — same hues, more
// saturation so dots pop on white.
const KIND_COLOR: Record<string, string> = {
  Book: "#2563eb",
  Chapter: "#93c5fd",
  Verse: "#cbd5e1",
  Person: "#dc2626",
  Place: "#16a34a",
  Event: "#f59e0b",
  Theme: "#9333ea",
};

// Diameter (px) at default zoom. Larger nodes for higher-hierarchy kinds.
const KIND_SIZE: Record<string, number> = {
  Book: 56,
  Chapter: 36,
  Verse: 20,
  Person: 42,
  Place: 38,
  Event: 42,
  Theme: 34,
};

const EDGE_COLOR: Record<string, string> = {
  CONTAINS: "#cbd5e1",
  MENTIONS: "#fca5a5",
  EXPRESSES: "#c4b5fd",
  OCCURS_IN: "#fcd34d",
  INVOLVES: "#fdba74",
};

const KIND_LEGEND = [
  { kind: "Book", label: "Book" },
  { kind: "Chapter", label: "Chapter" },
  { kind: "Verse", label: "Verse" },
  { kind: "Person", label: "Person" },
  { kind: "Place", label: "Place" },
  { kind: "Event", label: "Event" },
  { kind: "Theme", label: "Theme" },
];

function nodeLabel(n: SubgraphNode): string {
  if (n.kind === "Verse") return n.ref ?? n.label ?? n.id;
  return n.label ?? n.name ?? n.id;
}

function toElements(subgraph: Subgraph): ElementDefinition[] {
  const els: ElementDefinition[] = [];
  for (const n of subgraph.nodes) {
    const kind = n.kind ?? "?";
    els.push({
      data: {
        id: n.id,
        label: nodeLabel(n),
        kind,
        color: KIND_COLOR[kind] ?? "#9ca3af",
        size: KIND_SIZE[kind] ?? 28,
        text: n.text ?? "",
        ref: n.ref ?? "",
      },
    });
  }
  const seen = new Set<string>();
  for (const l of subgraph.links) {
    const key = `${l.source}->${l.target}->${l.kind ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const kind = l.kind ?? "";
    els.push({
      data: {
        id: key,
        source: l.source,
        target: l.target,
        kind,
        color: EDGE_COLOR[kind] ?? "#e2e8f0",
      },
    });
  }
  return els;
}

export function GraphPanel({ subgraph }: { subgraph: Subgraph | null | undefined }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{ label: string; kind: string; text?: string } | null>(null);

  const elements = useMemo(
    () => (subgraph && subgraph.nodes.length > 0 ? toElements(subgraph) : []),
    [subgraph],
  );

  // Mount Cytoscape once.
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements: [],
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            label: "data(label)",
            "text-valign": "bottom",
            "text-halign": "center",
            "text-margin-y": 4,
            "font-size": 10,
            "font-weight": 500,
            color: "#1f2937",
            "text-outline-width": 2,
            "text-outline-color": "#ffffff",
            "text-outline-opacity": 0.9,
            width: "data(size)",
            height: "data(size)",
            "border-width": 1.5,
            "border-color": "#ffffff",
            "border-opacity": 1,
            "overlay-opacity": 0,
            "transition-property": "border-color, border-width, background-color",
            "transition-duration": 150,
          },
        },
        {
          selector: 'node[kind = "Verse"]',
          style: { "font-size": 9, color: "#475569" },
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": "data(color)",
            "target-arrow-color": "data(color)",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.7,
            "curve-style": "bezier",
            opacity: 0.55,
            "transition-property": "opacity, width, line-color",
            "transition-duration": 150,
          },
        },
        {
          selector: ".faded",
          style: { opacity: 0.08, "text-opacity": 0.08 },
        },
        {
          selector: ".highlight",
          style: { opacity: 1, "text-opacity": 1, "z-index": 999 },
        },
        {
          selector: "node.highlight",
          style: {
            "border-color": "#3b82f6",
            "border-width": 3,
          },
        },
        {
          selector: "edge.highlight",
          style: {
            width: 2.5,
            opacity: 0.9,
          },
        },
      ],
    });
    cyRef.current = cy;

    // Cytoscape renders into the container's current size at mount time;
    // when the flex layout settles (or the tab switches), we need to tell
    // cytoscape to resize + refit, otherwise it stays 0×0 forever.
    const ro = new ResizeObserver(() => {
      cy.resize();
      if (cy.elements().length > 0) cy.fit(undefined, 30);
    });
    ro.observe(containerRef.current);

    // Hover: highlight the neighborhood, fade the rest.
    cy.on("mouseover", "node", (e) => {
      const node = e.target;
      const nbhd = node.closedNeighborhood();
      cy.elements().addClass("faded");
      nbhd.removeClass("faded").addClass("highlight");
      setHoverInfo({
        label: node.data("label"),
        kind: node.data("kind"),
        text: node.data("text") || undefined,
      });
    });
    cy.on("mouseout", "node", () => {
      cy.elements().removeClass("faded").removeClass("highlight");
      setHoverInfo(null);
    });

    return () => {
      ro.disconnect();
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  // Whenever the subgraph changes, replace elements and re-run the layout.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().remove();
    if (elements.length === 0) return;
    cy.add(elements);
    cy.resize();
    const layout = cy.layout({
      name: "fcose",
      // @ts-expect-error — fcose options not in cytoscape's base types
      animate: true,
      animationDuration: 600,
      animationEasing: "ease-out",
      nodeRepulsion: () => 8000,
      idealEdgeLength: () => 90,
      edgeElasticity: () => 0.45,
      gravity: 0.25,
      padding: 30,
      randomize: true,
      nodeSeparation: 75,
      fit: true,
    });
    layout.run();
    cy.one("layoutstop", () => {
      cy.resize();
      cy.fit(undefined, 30);
    });
  }, [elements]);

  const isEmpty = !subgraph || subgraph.nodes.length === 0;

  return (
    <div className="relative flex-1 min-h-0 bg-white">
      {/* Always-mounted canvas so the cytoscape init effect can grab its
          ref on first render, even when the subgraph is empty. */}
      <div
        ref={containerRef}
        className="absolute inset-0"
        style={{ width: "100%", height: "100%", minHeight: 300 }}
      />

      {isEmpty && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500 pointer-events-none">
          Ask a question to see the answer&apos;s subgraph.
        </div>
      )}

      {/* Legend */}
      <div className="absolute top-2 left-2 bg-white/95 backdrop-blur rounded-md border border-gray-200 px-2.5 py-2 text-[10px] shadow-sm pointer-events-none">
        <div className="font-semibold text-gray-600 mb-1">Nodes</div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
          {KIND_LEGEND.map((l) => (
            <div key={l.kind} className="flex items-center gap-1.5">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{ background: KIND_COLOR[l.kind] }}
              />
              <span className="text-gray-700">{l.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Hover detail card */}
      {hoverInfo && (
        <div className="absolute bottom-2 left-2 right-2 max-w-md bg-white/95 backdrop-blur rounded-md border border-gray-200 px-3 py-2 text-xs shadow-sm pointer-events-none">
          <div className="flex items-center gap-2 mb-0.5">
            <span
              className="inline-block w-2 h-2 rounded-full"
              style={{ background: KIND_COLOR[hoverInfo.kind] ?? "#9ca3af" }}
            />
            <span className="font-semibold text-gray-900">{hoverInfo.label}</span>
            <span className="text-gray-400">· {hoverInfo.kind}</span>
          </div>
          {hoverInfo.text && (
            <div className="text-gray-700 leading-snug line-clamp-3">{hoverInfo.text}</div>
          )}
        </div>
      )}
    </div>
  );
}
