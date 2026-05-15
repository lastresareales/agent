import { useEffect, useMemo, useState } from "react";

const sampleText = "Linus Torvalds created Linux in Helsinki. Sundar Pichai leads Google from Mountain View.";

const labelMeta = {
  PER: { label: "Person", color: "#2563eb" },
  ORG: { label: "Organization", color: "#0f766e" },
  LOC: { label: "Location", color: "#b45309" },
  MISC: { label: "Misc", color: "#7c3aed" },
};

function renderHighlightedText(text, entities) {
  const parts = [];
  let cursor = 0;

  for (const entity of entities) {
    if (entity.start > cursor) {
      parts.push(text.slice(cursor, entity.start));
    }

    parts.push(
      <mark
        key={entity.id}
        className="entity-highlight"
        style={{ "--entity-color": labelMeta[entity.label].color }}
      >
        {entity.text}
        <span>{entity.label}</span>
      </mark>,
    );
    cursor = entity.end;
  }

  if (cursor < text.length) {
    parts.push(text.slice(cursor));
  }

  return parts;
}

function App() {
  const [text, setText] = useState(sampleText);
  const [result, setResult] = useState({ engine: "pending", entities: [], graph: { nodes: [], edges: [] } });
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [chatInput, setChatInput] = useState("Satya Nadella leads Microsoft from Redmond.");
  const [messages, setMessages] = useState([
    {
      id: "assistant-welcome",
      role: "assistant",
      text: "I can talk now. Tell me facts and I will store the relationships I can extract.",
      backend: "rules",
    },
  ]);
  const [chatStatus, setChatStatus] = useState("idle");

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      const trimmedText = text.trim();

      if (!trimmedText) {
        setResult({ engine: "empty", entities: [], graph: { nodes: [], edges: [] } });
        setStatus("idle");
        setError("");
        return;
      }

      setStatus("loading");
      setError("");

      try {
        const response = await fetch("/api/extract", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, learn: false }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`API request failed with ${response.status}`);
        }

        const payload = await response.json();
        setResult(payload);
        setStatus("ready");
      } catch (requestError) {
        if (requestError.name !== "AbortError") {
          setStatus("error");
          setError(requestError.message);
        }
      }
    }, 350);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [text]);

  const entities = useMemo(() => result.entities ?? [], [result.entities]);
  const graphEdges = result.graph?.edges ?? [];
  const engineLabel = result.engine === "bert" ? "BERT model" : "Fallback rules";

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="intro">
          <p className="eyebrow">Learning NER Assistant</p>
          <h1>Memory Chat</h1>
          <p>
            Talk with the assistant, learn facts into memory, and inspect the extraction pipeline behind each response.
          </p>
        </div>

        <section className="chat-panel" aria-label="Assistant chat">
          <div className="panel-heading">
            <h2>Conversation</h2>
            <span>{messages.length}</span>
          </div>
          <div className="message-list">
            {messages.map((message) => (
              <div className={`message message-${message.role}`} key={message.id}>
                <span>{message.role === "assistant" ? "Assistant" : "You"}</span>
                {message.backend ? <small>{message.backend}</small> : null}
                <p>{message.text}</p>
              </div>
            ))}
          </div>
          <form
            className="chat-form"
            onSubmit={async (event) => {
              event.preventDefault();
              const trimmed = chatInput.trim();
              if (!trimmed || chatStatus === "loading") {
                return;
              }

              const userMessage = { id: `user-${Date.now()}`, role: "user", text: trimmed };
              setMessages((currentMessages) => [...currentMessages, userMessage]);
              setChatInput("");
              setChatStatus("loading");

              try {
                const response = await fetch("/api/chat", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ message: trimmed }),
                });

                if (!response.ok) {
                  throw new Error(`Chat request failed with ${response.status}`);
                }

                const payload = await response.json();
                setResult(payload.extraction);
                setText(trimmed);
                setMessages((currentMessages) => [
                  ...currentMessages,
                  {
                    id: `assistant-${Date.now()}`,
                    role: "assistant",
                    text: payload.reply,
                    backend: payload.backend,
                  },
                ]);
                setChatStatus("ready");
              } catch (chatError) {
                setMessages((currentMessages) => [
                  ...currentMessages,
                  { id: `assistant-error-${Date.now()}`, role: "assistant", text: chatError.message },
                ]);
                setChatStatus("error");
              }
            }}
          >
            <textarea
              aria-label="Chat message"
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              spellCheck="true"
            />
            <button type="submit" disabled={chatStatus === "loading"}>
              {chatStatus === "loading" ? "Thinking" : "Send"}
            </button>
          </form>
        </section>

        <div className="editor-panel">
          <label htmlFor="text-input">Input text</label>
          <textarea
            id="text-input"
            value={text}
            onChange={(event) => setText(event.target.value)}
            spellCheck="false"
          />
          <div className="status-row" role="status">
            <span className={`status-dot status-${status}`} />
            <span>{status === "loading" ? "Analyzing text" : `Engine: ${engineLabel}`}</span>
            {error ? <strong>{error}</strong> : null}
          </div>
        </div>

        <section className="results-grid" aria-label="Entity recognition results">
          <div className="result-panel">
            <div className="panel-heading">
              <h2>Detected Entities</h2>
              <span>{entities.length}</span>
            </div>
            <div className="highlight-box">{renderHighlightedText(text, entities)}</div>
          </div>

          <div className="result-panel">
            <div className="panel-heading">
              <h2>Knowledge Graph</h2>
              <span>{graphEdges.length}</span>
            </div>
            <div className="graph-list">
              {graphEdges.length > 0 ? (
                graphEdges.map((edge) => (
                  <div className="graph-edge" key={`${edge.source}-${edge.relation}-${edge.target}`}>
                    <strong>{edge.source}</strong>
                    <span>{edge.relation}</span>
                    <strong>{edge.target}</strong>
                  </div>
                ))
              ) : (
                <p className="empty-state">No relationships returned for the current text.</p>
              )}
            </div>
          </div>
        </section>

        <section className="entity-table" aria-label="Extracted entity table">
          <div className="panel-heading">
            <h2>Entity Table</h2>
            <span>{new Set(entities.map((entity) => entity.label)).size} labels</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Text</th>
                  <th>Label</th>
                  <th>Type</th>
                </tr>
              </thead>
              <tbody>
                {entities.map((entity) => (
                  <tr key={entity.id}>
                    <td>{entity.text}</td>
                    <td>
                      <span className="label-pill" style={{ "--entity-color": labelMeta[entity.label].color }}>
                        {entity.label}
                      </span>
                    </td>
                    <td>{labelMeta[entity.label].label}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </section>
    </main>
  );
}

export default App;
