import { useEffect, useRef, useState } from 'react';
import { streamConversationalRecommendations } from '../api/recommenderApi';
import styles from './Chatbot.module.css';

function renderMarkdown(text) {
  return text
    .split('\n')
    .filter(Boolean)
    .map((line, index) => {
      if (line.startsWith('- ')) return <li key={index}>{line.slice(2)}</li>;
      return <p key={index}>{line}</p>;
    });
}

export default function Chatbot({ userId }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: '✨ Welcome to RoomSense.\nYour smart furniture assistant for discovering stylish, functional, and personalized interiors.\n\nAsk me anything — from modern sofas to complete room setups, and I’ll find the best matches for you.',
    },
  ]);
  const [streaming, setStreaming] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages, streaming]);

  function appendAssistantToken(token) {
    setMessages((current) => {
      const next = [...current];
      const last = next[next.length - 1];

      if (last?.role === 'assistant' && last.streaming) {
        next[next.length - 1] = { ...last, content: `${last.content}${token}` };
      } else {
        next.push({ role: 'assistant', content: token, streaming: true });
      }

      return next;
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const query = input.trim();
    if (!query || streaming) return;

    setInput('');
    setStreaming(true);
    setMessages((current) => [
      ...current,
      { role: 'user', content: query },
      { role: 'assistant', content: '', streaming: true, products: [] },
    ]);

    try {
      await streamConversationalRecommendations({
        user_id: userId,
        query,
        session_id: sessionId,
        top_k: 10,
        onProducts: ({ products, show_products, session_id }) => {
          setSessionId(session_id);
          setMessages((current) => {
            const next = [...current];
            const last = next[next.length - 1];
            next[next.length - 1] = {
              ...last,
              products: show_products ? products : [],
            };
            return next;
          });
        },
        onToken: appendAssistantToken,
        onDone: ({ session_id }) => {
          setSessionId(session_id);
          setMessages((current) => current.map((message) => (
            message.streaming ? { ...message, streaming: false } : message
          )));
        },
        onError: ({ message }) => {
          appendAssistantToken(message);
        },
      });
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: 'assistant', content: error.message || 'Chat request failed.' },
      ]);
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className={styles.chatRoot}>
      <button
        className={styles.fab}
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-label="Open AI shopping assistant"
      >
        AI
      </button>

      <section className={`${styles.panel} ${open ? styles.panelOpen : ''}`}>
        <header className={styles.header}>
          <div>
            <h2>RoomSense Assistant</h2>
            <span>Local Mistral + product search</span>
          </div>
          <button type="button" onClick={() => setOpen(false)} className={styles.closeBtn}>
            x
          </button>
        </header>

        <div className={styles.messages} ref={scrollRef}>
          {messages.map((message, index) => (
            <article
              key={`${message.role}-${index}`}
              className={`${styles.message} ${message.role === 'user' ? styles.user : styles.assistant}`}
            >
              <div className={styles.bubble}>
                {message.content ? renderMarkdown(message.content) : <span className={styles.cursor} />}
                {message.streaming && <span className={styles.cursor} />}
              </div>

              {message.products?.length > 0 && (
                <div className={styles.productStrip}>
                  {message.products.slice(0, 3).map((product) => (
                    <div key={product.product_id} className={styles.productMini}>
                      <strong>{product.product_name}</strong>
                      <span>{product.category}</span>
                    </div>
                  ))}
                </div>
              )}
            </article>
          ))}
        </div>

        <form className={styles.form} onSubmit={handleSubmit}>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Chat or ask for a sofa, desk, budget..."
            disabled={streaming}
          />
          <button type="submit" disabled={streaming || !input.trim()}>
            {streaming ? '...' : 'Send'}
          </button>
        </form>
      </section>
    </div>
  );
}
