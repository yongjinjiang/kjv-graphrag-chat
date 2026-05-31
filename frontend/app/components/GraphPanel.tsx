"use client";

import { useMemo } from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";

import "@xyflow/react/dist/style.css";

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

const KIND_COLOR: Record<string, string> = {
  Book: "#1f77b4",
  Chapter: "#aec7e8",
  Verse: "#e5e7eb",
  Person: "#d62728",
  Place: "#2ca02c",
  Event: "#ff7f0e",
  Theme: "#9467bd",
};

const KIND_TEXT: Record<string, string> = {
  Book: "#ffffff",
  Chapter: "#1f2937",
  Verse: "#1f2937",
  Person: "#ffffff",
  Place: "#ffffff",
  Event: "#ffffff",
  Theme: "#ffffff",
};

const KIND_WIDTH: Record<string, number> = {
  Book: 130,
  Chapter: 110,
  Verse: 90,
  Person: 120,
  Place: 120,
  Event: 130,
  Theme: 110,
};

const KIND_LEGEND: { kind: string; label: string }[] = [
  { kind: "Book", label: "Book" },
  { kind: "Chapter", label: "Chapter" },
  { kind: "Verse", label: "Verse" },
  { kind: "Person", label: "Person" },
  { kind: "Place", label: "Place" },
  { kind: "Event", label: "Event" },
  { kind: "Theme", label: "Theme" },
];

const NODE_HEIGHT = 36;

function nodeLabel(n: SubgraphNode): string {
  if (n.kind === "Verse") return n.ref ?? n.label ?? n.id;
  return n.label ?? n.name ?? n.id;
}

function nodeTitle(n: SubgraphNode): string {
  if (n.kind === "Verse" && n.text) {
    return `${n.ref ?? n.id}\n\n${n.text}`;
  }
  return `${n.kind ?? "?"}: ${nodeLabel(n)}`;
}

function computePositions(
  nodes: SubgraphNode[],
  links: SubgraphLink[],
): Map<string, { x: number; y: number }> {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "TB", nodesep: 28, ranksep: 60, marginx: 16, marginy: 16 });
  g.setDefaultEdgeLabel(() => ({}));
  for (const n of nodes) {
    g.setNode(n.id, {
      width: KIND_WIDTH[n.kind ?? ""] ?? 110,
      height: NODE_HEIGHT,
    });
  }
  const seen = new Set<string>();
  for (const l of links) {
    const key = `${l.source}->${l.target}`;
    if (seen.has(key)) continue;
    seen.add(key);
    g.setEdge(l.source, l.target);
  }
  dagre.layout(g);
  const out = new Map<string, { x: number; y: number }>();
  for (const n of nodes) {
    const p = g.node(n.id);
    if (p) out.set(n.id, { x: p.x - p.width / 2, y: p.y - p.height / 2 });
  }
  return out;
}

export function GraphPanel({ subgraph }: { subgraph: Subgraph | null | undefined }) {
  const { rfNodes, rfEdges } = useMemo(() => {
    if (!subgraph || subgraph.nodes.length === 0) {
      return { rfNodes: [] as Node[], rfEdges: [] as Edge[] };
    }
    const positions = computePositions(subgraph.nodes, subgraph.links);
    const nodes: Node[] = subgraph.nodes.map((n) => {
      const pos = positions.get(n.id) ?? { x: 0, y: 0 };
      const kind = n.kind ?? "";
      return {
        id: n.id,
        position: pos,
        data: { label: nodeLabel(n) },
        draggable: true,
        style: {
          background: KIND_COLOR[kind] ?? "#9ca3af",
          color: KIND_TEXT[kind] ?? "#ffffff",
          border: "1px solid rgba(0,0,0,0.08)",
          borderRadius: 6,
          padding: "6px 10px",
          fontSize: 11,
          fontWeight: 500,
          width: KIND_WIDTH[kind] ?? 110,
        },
        title: nodeTitle(n),
      } as Node;
    });

    const edgeSeen = new Set<string>();
    const edges: Edge[] = [];
    for (const l of subgraph.links) {
      const key = `${l.source}->${l.target}`;
      if (edgeSeen.has(key)) continue;
      edgeSeen.add(key);
      edges.push({
        id: key,
        source: l.source,
        target: l.target,
        label: l.kind,
        type: "smoothstep",
        style: { stroke: "#cbd5e1", strokeWidth: 1.2 },
        labelStyle: { fontSize: 9, fill: "#64748b" },
        labelBgStyle: { fill: "#ffffff", fillOpacity: 0.9 },
        labelBgPadding: [2, 2],
      });
    }

    return { rfNodes: nodes, rfEdges: edges };
  }, [subgraph]);

  if (!subgraph || subgraph.nodes.length === 0) {
    return (
      <div className="p-5 text-sm text-gray-500">
        Ask a question to see the answer&apos;s subgraph.
      </div>
    );
  }

  return (
    <div className="flex-1 relative min-h-0">
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.2}
        maxZoom={2}
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable
      >
        <Background gap={16} color="#e5e7eb" />
        <Controls showInteractive={false} />
        <MiniMap
          pannable
          zoomable
          nodeColor={(n) =>
            (n.style?.background as string | undefined) ?? "#9ca3af"
          }
          maskColor="rgba(255,255,255,0.6)"
        />
      </ReactFlow>
      <div className="absolute top-2 left-2 bg-white/90 backdrop-blur rounded-md border border-gray-200 px-2 py-1.5 text-[10px] space-y-0.5 shadow-sm">
        <div className="font-semibold text-gray-600 mb-0.5">Legend</div>
        {KIND_LEGEND.map((l) => (
          <div key={l.kind} className="flex items-center gap-1.5">
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm"
              style={{ background: KIND_COLOR[l.kind] }}
            />
            <span className="text-gray-700">{l.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
