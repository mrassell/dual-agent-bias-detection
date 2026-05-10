import { useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || "";
const TABS = ["analyze", "dashboard", "stability", "verification", "logs"];
const DEMO_SENTENCES = [
  "Officials insisted that the policy was a complete success despite visible setbacks.",
  "According to court records, the proposal failed to receive enough votes.",
  "The senator boldly crushed opponents with a stunning and undeniable argument.",
  "The report cites multiple sources and avoids emotionally charged language."
];
const AUDITOR_MODEL_OPTIONS = ["gpt-4o-mini"];
const VERIFIER_MODEL_OPTIONS = ["claude-haiku-4-5-20251001"];
const BASELINE_MODEL = "roberta-babe-basil-ft";

function scoreSentence(sentence) {
  const emotionalWords = [
    "boldly",
    "stunning",
    "undeniable",
    "insisted",
    "crushed",
    "complete",
    "visible",
    "outrageous",
    "shocking",
    "destroyed"
  ];
  const loadedHits = emotionalWords.filter((word) =>
    sentence.toLowerCase().includes(word)
  ).length;
  const lexicalScore = Math.min(1, loadedHits * 0.18);
  const contextualScore = sentence.toLowerCase().includes("according to") ? 0.25 : 0.62;
  const severity = (lexicalScore + contextualScore) / 2;
  return {
    sentence,
    lexical_score: Number(lexicalScore.toFixed(2)),
    contextual_score: Number(contextualScore.toFixed(2)),
    overall_bias_score: Number(severity.toFixed(2)),
    likely_bias: severity >= 0.5
  };
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function splitToSentences(rawText) {
  return rawText
    .split(/[.!?]\s+/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function severityBand(score) {
  if (score < 0.34) return "Low";
  if (score < 0.67) return "Medium";
  return "High";
}

function downloadFile(content, fileName, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [auditorModelName] = useState(AUDITOR_MODEL_OPTIONS[0]);
  const [verifierModelName] = useState(VERIFIER_MODEL_OPTIONS[0]);
  const [enableStability, setEnableStability] = useState(true);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [text, setText] = useState(DEMO_SENTENCES.join(" "));
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [toolLogs, setToolLogs] = useState([
    "[12:03:08] session_started: demo pipeline initialized",
    `[12:03:10] models_loaded: auditor=${AUDITOR_MODEL_OPTIONS[0]}, verifier=${VERIFIER_MODEL_OPTIONS[0]}, baseline=${BASELINE_MODEL}`,
    "[12:03:15] waiting_for_input: ready"
  ]);

  const modeLabel = useMemo(
    () => (API_BASE_URL ? `API mode (${API_BASE_URL})` : "Demo mode (no backend)"),
    []
  );

  const summary = useMemo(() => {
    if (results.length === 0) {
      return {
        sentenceCount: 0,
        avgOverall: 0,
        likelyBiasCount: 0,
        highRiskShare: 0
      };
    }
    const totalOverall = results.reduce(
      (accumulator, row) => accumulator + Number(row.overall_bias_score || 0),
      0
    );
    const likelyBiasCount = results.filter((row) => row.likely_bias).length;
    return {
      sentenceCount: results.length,
      avgOverall: totalOverall / results.length,
      likelyBiasCount,
      highRiskShare: likelyBiasCount / results.length
    };
  }, [results]);

  const distribution = useMemo(() => {
    const counts = { Low: 0, Medium: 0, High: 0 };
    results.forEach((row) => {
      counts[severityBand(Number(row.overall_bias_score || 0))] += 1;
    });
    return counts;
  }, [results]);

  const heatmapCells = useMemo(() => {
    const rows = results.slice(0, 12).map((row, index) => {
      const score = Number(row.overall_bias_score || 0);
      return {
        id: `${row.sentence}-${index}`,
        label: `S${index + 1}`,
        score
      };
    });
    return rows;
  }, [results]);

  const stabilitySeries = useMemo(() => {
    const base = summary.avgOverall || 0.25;
    return [0, 1, 2, 3, 4, 5].map((point) => ({
      run: point + 1,
      drift: Number(Math.max(0, Math.min(1, base + (point - 2) * 0.06)).toFixed(2))
    }));
  }, [summary.avgOverall]);

  const verificationRows = useMemo(
    () =>
      results.slice(0, 8).map((row, index) => ({
        id: index + 1,
        claim: row.sentence,
        support_confidence: Number(
          Math.max(0.1, 1 - Number(row.overall_bias_score || 0)).toFixed(2)
        ),
        verdict: row.likely_bias ? "Needs review" : "Supported"
      })),
    [results]
  );

  const runAnalysis = async () => {
    setError("");
    setStatus("running");

    const sentences = splitToSentences(text);
    if (sentences.length === 0) {
      setStatus("idle");
      setError("Add at least one sentence before running analysis.");
      return;
    }

    setToolLogs((previous) => [
      ...previous,
      `[${new Date().toLocaleTimeString()}] detect_bias_started: ${sentences.length} sentence(s), auditor=${auditorModelName}, verifier=${verifierModelName}, baseline=${BASELINE_MODEL}`,
      enableStability
        ? `[${new Date().toLocaleTimeString()}] stability_enabled: running drift probes`
        : `[${new Date().toLocaleTimeString()}] stability_disabled: skipped`
    ]);

    if (!API_BASE_URL) {
      const localResults = sentences.map(scoreSentence);
      setResults(localResults);
      setStatus("done");
      setToolLogs((previous) => [
        ...previous,
        `[${new Date().toLocaleTimeString()}] detect_bias_finished: ${localResults.length} result(s) produced`
      ]);
      return;
    }

    try {
      const responses = await Promise.all(
        sentences.map(async (sentence) => {
          const response = await fetch(`${API_BASE_URL}/detect_bias`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              sentence,
              auditor_model: auditorModelName,
              verifier_model: verifierModelName,
              baseline_model: BASELINE_MODEL,
              stability: enableStability
            })
          });
          if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
          }
          return response.json();
        })
      );
      setResults(responses);
      setStatus("done");
      setToolLogs((previous) => [
        ...previous,
        `[${new Date().toLocaleTimeString()}] detect_bias_finished: API results received`
      ]);
    } catch (err) {
      setStatus("idle");
      setError(err.message || "Unknown error while calling API.");
      setToolLogs((previous) => [
        ...previous,
        `[${new Date().toLocaleTimeString()}] error: ${err.message || "unknown"}`
      ]);
    }
  };

  const loadSample = () => {
    setText(DEMO_SENTENCES.join(" "));
    setError("");
  };

  const clearAll = () => {
    setText("");
    setResults([]);
    setError("");
    setStatus("idle");
  };

  const onFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploadedFileName(file.name);
    const content = await file.text();
    setText(content);
    setToolLogs((previous) => [
      ...previous,
      `[${new Date().toLocaleTimeString()}] corpus_uploaded: ${file.name} (${content.length} chars)`
    ]);
  };

  const exportResultsJson = () => {
    const payload = {
      models: {
        auditor: auditorModelName,
        verifier: verifierModelName,
        baseline: BASELINE_MODEL
      },
      stability_enabled: enableStability,
      summary,
      results
    };
    downloadFile(
      JSON.stringify(payload, null, 2),
      "bias-analysis-results.json",
      "application/json"
    );
  };

  const exportResultsCsv = () => {
    const header = [
      "sentence",
      "lexical_score",
      "contextual_score",
      "overall_bias_score",
      "likely_bias"
    ];
    const rows = results.map((row) =>
      [
        JSON.stringify(row.sentence ?? ""),
        row.lexical_score ?? "",
        row.contextual_score ?? "",
        row.overall_bias_score ?? "",
        row.likely_bias ?? ""
      ].join(",")
    );
    downloadFile(
      [header.join(","), ...rows].join("\n"),
      "bias-analysis-results.csv",
      "text/csv;charset=utf-8;"
    );
  };

  const exportLogs = () => {
    downloadFile(toolLogs.join("\n"), "mcp-tool-logs.txt", "text/plain;charset=utf-8;");
  };

  return (
    <main className="page">
      <section className="hero">
        <div className="hero-top">
          <div>
            <h1>Dual-Agent Bias Detection</h1>
            <p>Demo cockpit for analysis, drift stability, verification, and audit logs.</p>
          </div>
          <span className="mode-pill">{modeLabel}</span>
        </div>
        <div className="stats-grid">
          <article className="stat-card">
            <p className="stat-label">Sentences Scored</p>
            <p className="stat-value">{summary.sentenceCount}</p>
          </article>
          <article className="stat-card">
            <p className="stat-label">Average Bias Score</p>
            <p className="stat-value">{summary.avgOverall.toFixed(2)}</p>
          </article>
          <article className="stat-card">
            <p className="stat-label">Likely Bias Flags</p>
            <p className="stat-value">{summary.likelyBiasCount}</p>
          </article>
        </div>
      </section>

      <section className="tabs" role="tablist" aria-label="Demo sections">
        {TABS.map((tab) => {
          const isDisabled = tab === "stability" && !enableStability;
          return (
            <button
              key={tab}
              className={`tab-btn ${activeTab === tab ? "active" : ""}`}
              onClick={() => setActiveTab(tab)}
              disabled={isDisabled}
            >
              {tab === "analyze" && "Analyze"}
              {tab === "dashboard" && "Dashboard"}
              {tab === "stability" && "Stability Panel"}
              {tab === "verification" && "Verification"}
              {tab === "logs" && "Tool Logs"}
            </button>
          );
        })}
      </section>

      {activeTab === "analyze" ? (
        <section className="panel">
          <div className="panel-grid">
            <div>
              <h2>Upload Panel</h2>
              <p className="subdued">Upload text corpus</p>
              <input type="file" accept=".txt,.md,.csv,.jsonl,.json" onChange={onFileUpload} />
              <p className="subdued">
                {uploadedFileName ? `Loaded: ${uploadedFileName}` : "No corpus uploaded yet."}
              </p>
            </div>
            <div>
              <h2>Model Settings</h2>
              <label htmlFor="auditor-model-select">Auditor Model</label>
              <select
                id="auditor-model-select"
                value={auditorModelName}
                disabled
              >
                {AUDITOR_MODEL_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              <label htmlFor="verifier-model-select">Verifier Model</label>
              <select
                id="verifier-model-select"
                value={verifierModelName}
                disabled
              >
                {VERIFIER_MODEL_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              <label>Baseline Comparison</label>
              <p className="subdued baseline-model">{BASELINE_MODEL}</p>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={enableStability}
                  onChange={(event) => setEnableStability(event.target.checked)}
                />
                Enable stability testing
              </label>
            </div>
          </div>

          <label htmlFor="article-input">Input text</label>
          <textarea
            id="article-input"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Paste article text here..."
          />

          <div className="actions">
            <button onClick={runAnalysis} disabled={status === "running"}>
              {status === "running" ? "Analyzing..." : "Run Analysis"}
            </button>
            <button className="secondary-btn" onClick={loadSample}>
              Load Demo Sample
            </button>
            <button className="ghost-btn" onClick={clearAll}>
              Clear
            </button>
          </div>
          <div className="export-actions">
            <button className="ghost-btn" onClick={exportResultsJson} disabled={results.length === 0}>
              Export Results JSON
            </button>
            <button className="ghost-btn" onClick={exportResultsCsv} disabled={results.length === 0}>
              Export Results CSV
            </button>
          </div>

          {error ? <p className="error">{error}</p> : null}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Sentence</th>
                  <th>Lexical</th>
                  <th>Context</th>
                  <th>Overall</th>
                  <th>Flag</th>
                </tr>
              </thead>
              <tbody>
                {results.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="placeholder">
                      Run analysis to generate scores.
                    </td>
                  </tr>
                ) : (
                  results.map((row, index) => (
                    <tr key={`${row.sentence}-${index}`}>
                      <td>{row.sentence}</td>
                      <td>{row.lexical_score ?? "-"}</td>
                      <td>{row.contextual_score ?? "-"}</td>
                      <td>{row.overall_bias_score ?? "-"}</td>
                      <td>
                        <span
                          className={
                            row.likely_bias ? "status-chip danger" : "status-chip safe"
                          }
                        >
                          {row.likely_bias ? "Likely bias" : "Low bias"}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {activeTab === "dashboard" ? (
        <section className="panel">
          <div className="results-header">
            <h2>Dashboard</h2>
            <p>Bias distribution and heatmap</p>
          </div>
          <div className="dashboard-grid">
            <article className="card">
              <h3>Bias Distribution</h3>
              {Object.entries(distribution).map(([label, count]) => {
                const percent = results.length ? count / results.length : 0;
                return (
                  <div key={label} className="bar-row">
                    <span>{label}</span>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: formatPercent(percent) }} />
                    </div>
                    <strong>{count}</strong>
                  </div>
                );
              })}
            </article>
            <article className="card">
              <h3>Bias Heatmap</h3>
              <div className="heatmap-grid">
                {heatmapCells.length === 0 ? (
                  <p className="subdued">Run analysis to populate heatmap.</p>
                ) : (
                  heatmapCells.map((cell) => (
                    <div
                      key={cell.id}
                      className="heat-cell"
                      style={{ opacity: Math.max(0.2, cell.score) }}
                      title={`${cell.label}: ${cell.score}`}
                    >
                      {cell.label}
                    </div>
                  ))
                )}
              </div>
            </article>
          </div>
        </section>
      ) : null}

      {activeTab === "stability" && enableStability ? (
        <section className="panel">
          <div className="results-header">
            <h2>Stability Panel</h2>
            <p>Drift timeline</p>
          </div>
          <div className="timeline">
            {stabilitySeries.map((point) => (
              <div key={point.run} className="timeline-row">
                <span>Run {point.run}</span>
                <div className="bar-track">
                  <div className="bar-fill drift" style={{ width: formatPercent(point.drift) }} />
                </div>
                <strong>{point.drift.toFixed(2)}</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {activeTab === "verification" ? (
        <section className="panel">
          <div className="results-header">
            <h2>Verification Tab</h2>
            <p>Secondary verifier confidence view</p>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Claim</th>
                  <th>Support Confidence</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {verificationRows.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="placeholder">
                      No verification rows yet. Run analysis first.
                    </td>
                  </tr>
                ) : (
                  verificationRows.map((row) => (
                    <tr key={row.id}>
                      <td>#{row.id}</td>
                      <td>{row.claim}</td>
                      <td>{row.support_confidence}</td>
                      <td>{row.verdict}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {activeTab === "logs" ? (
        <section className="panel">
          <div className="results-header">
            <h2>Tool Log Viewer</h2>
            <p>MCP execution trace logs</p>
          </div>
          <div className="export-actions">
            <button className="ghost-btn" onClick={exportLogs} disabled={toolLogs.length === 0}>
              Export Logs
            </button>
          </div>
          <div className="log-viewer">
            {toolLogs.map((line, index) => (
              <p key={`${line}-${index}`}>{line}</p>
            ))}
          </div>
        </section>
      ) : null}
    </main>
  );
}

export default App;
