import { useState, useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Colour helpers ──────────────────────────────────────────────
const riskColor = (prob) => {
  if (prob > 0.6) return "#E24B4A";
  if (prob > 0.35) return "#EF9F27";
  return "#1D9E75";
};
const riskLabel = (prob) =>
  prob > 0.6 ? "High" : prob > 0.35 ? "Medium" : "Low";

// ── Stat card ────────────────────────────────────────────────────
function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: "var(--bg2, #f5f5f3)",
      borderRadius: 10,
      padding: "14px 18px",
      minWidth: 130,
    }}>
      <div style={{ fontSize: 12, color: "#888", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 500, color: color || "inherit" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "#aaa", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ── D3 Graph ─────────────────────────────────────────────────────
function GraphViz({ nodes, edges, onSelectNode, selectedId }) {
  const svgRef = useRef(null);
  const simRef = useRef(null);

  useEffect(() => {
    if (!nodes.length) return;
    const el = svgRef.current;
    const W = el.clientWidth || 640, H = 380;

    d3.select(el).selectAll("*").remove();
    const svg = d3.select(el)
      .attr("viewBox", `0 0 ${W} ${H}`)
      .style("width", "100%").style("height", H);

    const g = svg.append("g");

    svg.call(d3.zoom()
      .scaleExtent([0.3, 4])
      .on("zoom", (e) => g.attr("transform", e.transform))
    );

    const link = g.append("g")
      .selectAll("line")
      .data(edges)
      .join("line")
      .attr("stroke", "#ddd")
      .attr("stroke-width", 0.6)
      .attr("stroke-opacity", 0.7);

    const node = g.append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => d.id === selectedId ? 9 : 6)
      .attr("fill", (d) => riskColor(d.churn_prob))
      .attr("stroke", (d) => d.id === selectedId ? "#222" : "none")
      .attr("stroke-width", 2)
      .attr("opacity", 0.85)
      .style("cursor", "pointer")
      .on("click", (_, d) => onSelectNode(d.id))
      .call(d3.drag()
        .on("start", (e, d) => { if (!e.active) simRef.current.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag",  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end",   (e, d) => { if (!e.active) simRef.current.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    node.append("title").text(d => `Customer ${d.id}\nChurn prob: ${(d.churn_prob * 100).toFixed(1)}%`);

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(edges).id(d => d.id).distance(40))
      .force("charge", d3.forceManyBody().strength(-60))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .on("tick", () => {
        link
          .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        node.attr("cx", d => d.x).attr("cy", d => d.y);
      });

    simRef.current = sim;
    return () => sim.stop();
  }, [nodes, edges, selectedId]);

  return <svg ref={svgRef} style={{ width: "100%", height: 380, display: "block" }} />;
}

// ── Main App ──────────────────────────────────────────────────────
export default function App() {
  const [stats, setStats]     = useState(null);
  const [graphData, setGraph] = useState({ nodes: [], edges: [] });
  const [topRisk, setTopRisk] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/graph-stats`).then(r => r.json()),
      fetch(`${API}/graph-sample?size=120`).then(r => r.json()),
      fetch(`${API}/top-at-risk?n=10`).then(r => r.json()),
    ])
      .then(([s, g, t]) => { setStats(s); setGraph(g); setTopRisk(t); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const handleSelectNode = useCallback((id) => {
    fetch(`${API}/customer/${id}`)
      .then(r => r.json())
      .then(setSelected);
  }, []);

  if (loading) return (
    <div style={{ padding: 40, color: "#888", textAlign: "center", fontSize: 15 }}>
      Loading model data...
    </div>
  );

  if (error) return (
    <div style={{ padding: 40, color: "#E24B4A", fontSize: 14 }}>
      Could not reach API at <code>{API}</code>. Make sure the Render backend is running.<br />
      <span style={{ color: "#888", fontSize: 12 }}>{error}</span>
    </div>
  );

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "24px 16px", fontFamily: "system-ui, sans-serif" }}>

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 500, margin: "0 0 4px" }}>
          GNN Churn Prediction
        </h1>
        <p style={{ fontSize: 13, color: "#888", margin: 0 }}>
          GraphSAGE · PyTorch Geometric · Telco dataset · {stats?.num_nodes?.toLocaleString()} customers
        </p>
      </div>

      {/* Stats row */}
      {stats && (
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 24 }}>
          <StatCard label="GNN AUC"   value={(stats.gnn_test_auc * 100).toFixed(1) + "%"} color="#534AB7" />
          <StatCard label="XGB AUC"   value={(stats.xgb_test_auc * 100).toFixed(1) + "%"} sub="baseline" />
          <StatCard label="Graph lift" value={"+" + ((stats.gnn_test_auc - stats.xgb_test_auc) * 100).toFixed(1) + "%"} color="#1D9E75" />
          <StatCard label="Churn rate" value={(stats.churn_rate * 100).toFixed(0) + "%"}   />
          <StatCard label="Edges"      value={stats.num_edges?.toLocaleString()} sub="call relationships" />
          <StatCard label="Avg degree" value={stats.avg_degree} sub="neighbors/customer" />
        </div>
      )}

      {/* Graph + sidebar */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 16, marginBottom: 24 }}>

        {/* D3 graph */}
        <div style={{ border: "0.5px solid #e0e0e0", borderRadius: 10, padding: 12, overflow: "hidden" }}>
          <div style={{ fontSize: 12, color: "#888", marginBottom: 8, display: "flex", gap: 16, alignItems: "center" }}>
            <span style={{ fontWeight: 500, color: "#333" }}>Customer social graph</span>
            <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#E24B4A", display: "inline-block" }} /> High risk
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#EF9F27", display: "inline-block" }} /> Medium
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#1D9E75", display: "inline-block" }} /> Low
            </span>
          </div>
          <GraphViz
            nodes={graphData.nodes}
            edges={graphData.edges}
            onSelectNode={handleSelectNode}
            selectedId={selected?.customer_id}
          />
          <div style={{ fontSize: 11, color: "#bbb", marginTop: 6 }}>
            Click a node to inspect · Drag to reposition · Scroll to zoom
          </div>
        </div>

        {/* Selected customer panel */}
        <div style={{ border: "0.5px solid #e0e0e0", borderRadius: 10, padding: 14 }}>
          {selected ? (
            <>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>
                Customer #{selected.customer_id}
              </div>

              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, color: "#888" }}>Churn probability</div>
                <div style={{ fontSize: 28, fontWeight: 500, color: riskColor(selected.churn_prob) }}>
                  {(selected.churn_prob * 100).toFixed(1)}%
                </div>
                <div style={{
                  display: "inline-block", fontSize: 11, padding: "2px 8px",
                  borderRadius: 20, marginTop: 4,
                  background: riskColor(selected.churn_prob) + "22",
                  color: riskColor(selected.churn_prob)
                }}>
                  {riskLabel(selected.churn_prob)} risk
                </div>
              </div>

              <div style={{ fontSize: 12, color: "#888", marginBottom: 6 }}>
                Tenure: {selected.tenure} months · ${selected.monthly_charges?.toFixed(0)}/mo
              </div>

              {selected.neighbors?.length > 0 && (
                <>
                  <div style={{ fontSize: 11, color: "#888", marginTop: 12, marginBottom: 6 }}>
                    Neighbors (avg churn: {(selected.avg_neighbor_churn_prob * 100).toFixed(0)}%)
                  </div>
                  {selected.neighbors.slice(0, 6).map(nb => (
                    <div
                      key={nb.id}
                      onClick={() => handleSelectNode(nb.id)}
                      style={{
                        display: "flex", justifyContent: "space-between",
                        padding: "5px 0", borderBottom: "0.5px solid #f0f0f0",
                        cursor: "pointer", fontSize: 12
                      }}
                    >
                      <span style={{ color: "#555" }}>#{nb.id}</span>
                      <span style={{ color: riskColor(nb.churn_prob), fontWeight: 500 }}>
                        {(nb.churn_prob * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </>
              )}
            </>
          ) : (
            <div style={{ color: "#bbb", fontSize: 13, textAlign: "center", paddingTop: 60 }}>
              Click a node to inspect a customer
            </div>
          )}
        </div>
      </div>

      {/* Top at-risk table */}
      <div style={{ border: "0.5px solid #e0e0e0", borderRadius: 10, padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Top 10 at-risk customers</div>
        <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "#888" }}>
              {["ID", "Churn prob", "Risk", "Tenure (mo)", "Monthly $"].map(h => (
                <th key={h} style={{ textAlign: "left", paddingBottom: 8, fontWeight: 400 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {topRisk.map(r => (
              <tr
                key={r.customer_id}
                onClick={() => handleSelectNode(r.customer_id)}
                style={{ cursor: "pointer", borderTop: "0.5px solid #f5f5f5" }}
              >
                <td style={{ padding: "7px 0", color: "#555" }}>#{r.customer_id}</td>
                <td style={{ fontWeight: 500, color: riskColor(r.churn_prob) }}>
                  {(r.churn_prob * 100).toFixed(1)}%
                </td>
                <td>
                  <span style={{
                    fontSize: 11, padding: "1px 6px", borderRadius: 20,
                    background: riskColor(r.churn_prob) + "22",
                    color: riskColor(r.churn_prob)
                  }}>
                    {riskLabel(r.churn_prob)}
                  </span>
                </td>
                <td style={{ color: "#555" }}>{r.tenure}</td>
                <td style={{ color: "#555" }}>${r.monthly_charges?.toFixed(0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 20, fontSize: 11, color: "#ccc", textAlign: "center" }}>
        GraphSAGE · PyTorch Geometric · FastAPI · React · D3.js · GitHub Pages · Render · Supabase
      </div>
    </div>
  );
}
