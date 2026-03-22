import { useState, useRef, useEffect } from "react";
import { Send, Bot, User } from "lucide-react";
import { sendChat } from "../api";

const THREAD_ID = `thread-${Date.now()}`;

export default function ChatInterface({ prescriptionContext }) {
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: "Hi! I can answer questions about this prescription. What would you like to know?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const msg = input.trim();
    if (!msg || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: msg }]);
    setInput("");
    setLoading(true);

    try {
      const res = await sendChat(msg, THREAD_ID, prescriptionContext);
      setMessages((prev) => [...prev, { role: "bot", text: res.data.reply }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Sorry, I couldn't process that. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <Bot size={18} />
        <span>Ask about this prescription</span>
      </div>

      <div className="chat-messages">
        {messages.map((m, i) => (
          <div key={i} className={`chat-bubble ${m.role === "user" ? "bubble-user" : "bubble-bot"}`}>
            <span className="bubble-icon">
              {m.role === "user" ? <User size={14} /> : <Bot size={14} />}
            </span>
            <p>{m.text}</p>
          </div>
        ))}
        {loading && (
          <div className="chat-bubble bubble-bot">
            <span className="bubble-icon"><Bot size={14} /></span>
            <p className="typing">Thinking...</p>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <textarea
          className="chat-input"
          rows={1}
          placeholder="Ask a question about the prescription..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
        />
        <button
          className="btn-send"
          onClick={send}
          disabled={loading || !input.trim()}
        >
          <Send size={16} />
        </button>
      </div>
    </div>
  );
}
