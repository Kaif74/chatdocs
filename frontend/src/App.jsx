import { useState } from "react";

import ChatPanel from "./components/ChatPanel";
import EvalPanel from "./components/EvalPanel";
import IngestPanel from "./components/IngestPanel";

const TABS = [
  { id: "chat", label: "Chat" },
  { id: "evaluate", label: "Evaluate" },
  { id: "ingest", label: "Ingest" },
];

function App() {
  const [activeTab, setActiveTab] = useState("chat");

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">CHATDOCS</p>
          <h1>Developer documentation QA with built-in evaluation</h1>
          <p className="hero__copy">
            Ingest LangChain, CrewAI, Next.js, and Expo docs into Chroma, answer
            developer questions, and score the system with both automatic
            metrics and human rubric inputs.
          </p>
        </div>
      </header>

      <nav className="tabs" aria-label="Application tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`tab ${activeTab === tab.id ? "tab--active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <section hidden={activeTab !== "chat"} aria-hidden={activeTab !== "chat"}>
        <ChatPanel />
      </section>

      <section
        hidden={activeTab !== "evaluate"}
        aria-hidden={activeTab !== "evaluate"}
      >
        <EvalPanel />
      </section>

      <section
        hidden={activeTab !== "ingest"}
        aria-hidden={activeTab !== "ingest"}
      >
        <IngestPanel />
      </section>
    </main>
  );
}

export default App;
