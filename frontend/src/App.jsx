import { useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || "";
const TABS = ["analyze", "dashboard", "baseline", "stability", "verification", "logs"];
const DEMO_SENTENCES = [
  "Officials insisted that the policy was a complete success despite visible setbacks.",
  "According to court records, the proposal failed to receive enough votes.",
  "The senator boldly crushed opponents with a stunning and undeniable argument.",
  "The report cites multiple sources and avoids emotionally charged language."
];
const PIPELINE_OPTIONS = [
  {
    id: "gpt_to_claude",
    label: "GPT -> Claude pipeline",
    auditor: "gpt-4o-mini",
    verifier: "claude-haiku-4-5-20251001"
  },
  {
    id: "claude_to_gpt",
    label: "Claude -> GPT pipeline",
    auditor: "claude-haiku-4-5-20251001",
    verifier: "gpt-4o-mini"
  },
  {
    id: "gpt_only",
    label: "GPT only",
    auditor: "gpt-4o-mini",
    verifier: "gpt-4o-mini"
  },
  {
    id: "claude_only",
    label: "Claude only",
    auditor: "claude-haiku-4-5-20251001",
    verifier: "claude-haiku-4-5-20251001"
  }
];
const BASELINE_MODEL = "roberta-babe-basil-ft";
const BIAS_RULES = [
  {
    type: "emotional",
    label: "Emotional language",
    severity: "high",
    terms: ["shocking", "outrageous", "destroyed", "crushed", "stunning"]
  },
  {
    type: "certainty",
    label: "Over-certainty",
    severity: "medium",
    terms: ["undeniable", "always", "never", "clearly", "obviously", "complete"]
  },
  {
    type: "framing",
    label: "Narrative framing",
    severity: "medium",
    terms: ["insisted", "admitted", "claimed", "slammed", "praised"]
  },
  {
    type: "speculation",
    label: "Speculative framing",
    severity: "low",
    terms: ["possibly", "appears", "seems", "likely", "reportedly"]
  }
];
const SEVERITY_WEIGHTS = { low: 0.09, medium: 0.16, high: 0.23 };
const RULE_ENTRIES = BIAS_RULES.flatMap((rule) =>
  rule.terms.map((term) => ({ ...rule, term }))
);
const TERM_META = new Map(
  RULE_ENTRIES.map((entry) => [
    entry.term.toLowerCase(),
    {
      type: entry.type,
      label: entry.label,
      severity: entry.severity
    }
  ])
);
const BIAS_TERM_PATTERN = RULE_ENTRIES.map((entry) =>
  entry.term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\\ /g, "\\s+")
)
  .sort((left, right) => right.length - left.length)
  .join("|");
const BIAS_TERM_REGEX = new RegExp(`\\b(${BIAS_TERM_PATTERN})\\b`, "gi");

function extractBiasMatches(text) {
  const matches = [];
  BIAS_TERM_REGEX.lastIndex = 0;
  let match;
  while ((match = BIAS_TERM_REGEX.exec(text)) !== null) {
    const matchedText = match[0];
    const metadata = TERM_META.get(matchedText.toLowerCase());
    if (!metadata) continue;
    matches.push({
      text: matchedText,
      start: match.index,
      end: match.index + matchedText.length,
      ...metadata
    });
  }
  return matches;
}

function scoreSentence(sentence) {
  const matches = extractBiasMatches(sentence);
  const weightedHits = matches.reduce(
    (accumulator, match) => accumulator + SEVERITY_WEIGHTS[match.severity],
    0
  );
  const lexicalScore = Math.min(1, weightedHits);
  const contextualScore = sentence.toLowerCase().includes("according to") ? 0.25 : 0.62;
  const severity = (lexicalScore + contextualScore) / 2;
  return {
    sentence,
    lexical_score: Number(lexicalScore.toFixed(2)),
    contextual_score: Number(contextualScore.toFixed(2)),
    overall_bias_score: Number(severity.toFixed(2)),
    likely_bias: severity >= 0.5,
    highlighted_terms: matches.map((match) => match.text),
    bias_types: [...new Set(matches.map((match) => match.type))]
  };
}

function applyPipelineAdjustment(scoredRow, pipelineId) {
  const adjustments = {
    gpt_to_claude: { lexical: 0, contextual: 0 },
    claude_to_gpt: { lexical: 0.03, contextual: -0.01 },
    gpt_only: { lexical: 0.05, contextual: 0.02 },
    claude_only: { lexical: -0.02, contextual: -0.03 }
  };
  const profile = adjustments[pipelineId] || adjustments.gpt_to_claude;
  const lexical = Math.max(0, Math.min(1, scoredRow.lexical_score + profile.lexical));
  const contextual = Math.max(0, Math.min(1, scoredRow.contextual_score + profile.contextual));
  const overall = (lexical + contextual) / 2;
  return {
    ...scoredRow,
    lexical_score: Number(lexical.toFixed(2)),
    contextual_score: Number(contextual.toFixed(2)),
    overall_bias_score: Number(overall.toFixed(2)),
    likely_bias: overall >= 0.5
  };
}

function baselineScoreSentence(sentence) {
  const base = scoreSentence(sentence);
  const lexical = Math.max(0, Math.min(1, base.lexical_score - 0.08));
  const contextual = Math.max(0, Math.min(1, base.contextual_score - 0.06));
  const overall = (lexical + contextual) / 2;
  return {
    ...base,
    lexical_score: Number(lexical.toFixed(2)),
    contextual_score: Number(contextual.toFixed(2)),
    overall_bias_score: Number(overall.toFixed(2)),
    likely_bias: overall >= 0.5
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

function highlightedSegments(text) {
  const matches = extractBiasMatches(text);
  if (matches.length === 0) return [{ text, plain: true }];

  const segments = [];
  let cursor = 0;
  matches.forEach((match) => {
    if (match.start > cursor) {
      segments.push({ text: text.slice(cursor, match.start), plain: true });
    }
    segments.push({
      text: text.slice(match.start, match.end),
      plain: false,
      type: match.type,
      label: match.label,
      severity: match.severity
    });
    cursor = match.end;
  });
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), plain: true });
  }
  return segments;
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
  const [pipelineId, setPipelineId] = useState(PIPELINE_OPTIONS[0].id);
  const [enableStability, setEnableStability] = useState(true);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [text, setText] = useState(DEMO_SENTENCES.join(" "));
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const activePipeline = useMemo(
    () => PIPELINE_OPTIONS.find((option) => option.id === pipelineId) ?? PIPELINE_OPTIONS[0],
    [pipelineId]
  );
  const [toolLogs, setToolLogs] = useState([
    "[12:03:08] session_started: demo pipeline initialized",
    `[12:03:10] pipeline_loaded: ${PIPELINE_OPTIONS[0].label}`,
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
  const textSegments = useMemo(() => highlightedSegments(text), [text]);
  const biasTypeCounts = useMemo(() => {
    const counts = {};
    textSegments.forEach((segment) => {
      if (segment.plain) return;
      counts[segment.label] = (counts[segment.label] || 0) + 1;
    });
    return counts;
  }, [textSegments]);
  const baselineComparisonRows = useMemo(() => {
    const sentences = splitToSentences(text);
    if (sentences.length === 0) return [];
    return sentences.map((sentence, index) => {
      const pipelineRow = results[index] ?? applyPipelineAdjustment(scoreSentence(sentence), pipelineId);
      const baselineRow = baselineScoreSentence(sentence);
      return {
        id: index + 1,
        sentence,
        pipeline_overall: Number(pipelineRow.overall_bias_score || 0),
        baseline_overall: Number(baselineRow.overall_bias_score || 0),
        delta: Number(
          (Number(pipelineRow.overall_bias_score || 0) - Number(baselineRow.overall_bias_score || 0)).toFixed(2)
        )
      };
    });
  }, [text, results, pipelineId]);
  const baselineSummary = useMemo(() => {
    if (baselineComparisonRows.length === 0) {
      return { pipelineAvg: 0, baselineAvg: 0, avgDelta: 0 };
    }
    const pipelineAvg =
      baselineComparisonRows.reduce((acc, row) => acc + row.pipeline_overall, 0) /
      baselineComparisonRows.length;
    const baselineAvg =
      baselineComparisonRows.reduce((acc, row) => acc + row.baseline_overall, 0) /
      baselineComparisonRows.length;
    return {
      pipelineAvg: Number(pipelineAvg.toFixed(2)),
      baselineAvg: Number(baselineAvg.toFixed(2)),
      avgDelta: Number((pipelineAvg - baselineAvg).toFixed(2))
    };
  }, [baselineComparisonRows]);

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
      `[${new Date().toLocaleTimeString()}] detect_bias_started: ${sentences.length} sentence(s), pipeline=${activePipeline.label}`,
      enableStability
        ? `[${new Date().toLocaleTimeString()}] stability_enabled: running drift probes`
        : `[${new Date().toLocaleTimeString()}] stability_disabled: skipped`
    ]);

    if (!API_BASE_URL) {
      const localResults = sentences.map((sentence) =>
        applyPipelineAdjustment(scoreSentence(sentence), pipelineId)
      );
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
              pipeline_id: activePipeline.id,
              auditor_model: activePipeline.auditor,
              verifier_model: activePipeline.verifier,
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
      pipeline: activePipeline,
      baseline_model: BASELINE_MODEL,
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
              {tab === "baseline" && "Baseline Compare"}
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
              <h2>Pipeline Settings</h2>
              <label htmlFor="pipeline-select">Select Pipeline</label>
              <select
                id="pipeline-select"
                value={pipelineId}
                onChange={(event) => setPipelineId(event.target.value)}
              >
                {PIPELINE_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
              <p className="subdued">
                Auditor: <strong>{activePipeline.auditor}</strong>
              </p>
              <p className="subdued">
                Verifier: <strong>{activePipeline.verifier}</strong>
              </p>
              <p className="subdued">
                Baseline comparison: <strong>{BASELINE_MODEL}</strong>
              </p>
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

          <div className="results-header annotated-head">
            <h3>Annotated Text View</h3>
            <p>Scroll and inspect highlighted bias terms by type and severity</p>
          </div>
          <div className="legend-row">
            {BIAS_RULES.map((rule) => (
              <span
                key={`${rule.type}-${rule.severity}`}
                className={`legend-chip type-${rule.type} severity-${rule.severity}`}
              >
                {rule.label}
              </span>
            ))}
          </div>
          <div className="legend-row counts-row">
            {Object.keys(biasTypeCounts).length === 0 ? (
              <span className="subdued">No highlighted bias terms detected yet.</span>
            ) : (
              Object.entries(biasTypeCounts).map(([label, count]) => (
                <span key={label} className="count-chip">
                  {label}: {count}
                </span>
              ))
            )}
          </div>
          <div className="annotated-text">
            {textSegments.map((segment, index) =>
              segment.plain ? (
                <span key={`${segment.text}-${index}`}>{segment.text}</span>
              ) : (
                <mark
                  key={`${segment.text}-${index}`}
                  className={`bias-highlight type-${segment.type} severity-${segment.severity}`}
                  title={`${segment.label} (${segment.severity})`}
                >
                  {segment.text}
                </mark>
              )
            )}
          </div>

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

      {activeTab === "baseline" ? (
        <section className="panel">
          <div className="results-header">
            <h2>Pipeline vs Baseline</h2>
            <p>
              Comparing selected pipeline against <code>{BASELINE_MODEL}</code>
            </p>
          </div>
          <div className="stats-grid">
            <article className="stat-card">
              <p className="stat-label">Pipeline Average</p>
              <p className="stat-value">{baselineSummary.pipelineAvg.toFixed(2)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Baseline Average</p>
              <p className="stat-value">{baselineSummary.baselineAvg.toFixed(2)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Average Delta</p>
              <p className="stat-value">{baselineSummary.avgDelta.toFixed(2)}</p>
            </article>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Sentence</th>
                  <th>Pipeline Overall</th>
                  <th>Baseline Overall</th>
                  <th>Delta</th>
                </tr>
              </thead>
              <tbody>
                {baselineComparisonRows.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="placeholder">
                      Add text in Analyze tab to see baseline comparison.
                    </td>
                  </tr>
                ) : (
                  baselineComparisonRows.map((row) => (
                    <tr key={row.id}>
                      <td>#{row.id}</td>
                      <td>{row.sentence}</td>
                      <td>{row.pipeline_overall.toFixed(2)}</td>
                      <td>{row.baseline_overall.toFixed(2)}</td>
                      <td>
                        <span className={row.delta >= 0 ? "delta-chip up" : "delta-chip down"}>
                          {row.delta >= 0 ? "+" : ""}
                          {row.delta.toFixed(2)}
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
