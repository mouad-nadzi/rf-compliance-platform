import React from 'react';
import { type ChatMessage as ChatMessageType } from '../../api';

interface Props {
  message: ChatMessageType;
  isStreaming?: boolean;
}

const renderFormattedContent = (content: string) => {
  if (!content) return null;

  const lines = content.split('\n');

  return lines.map((line, lineIdx) => {
    let trimmed = line.trim();
    if (!trimmed) {
      return <div key={lineIdx} style={{ height: '0.35rem' }} />;
    }

    const isBullet = /^\s*[\-\*]\s+/.test(line);
    if (isBullet) {
      trimmed = line.replace(/^\s*[\-\*]\s+/, '');
    }

    const parts = trimmed.split(/(\*\*.*?\*\*|\*.*?\*|`.*?`)/g);

    const formattedLine = parts.map((part, pIdx) => {
      if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
        return <strong key={pIdx} style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
        return <em key={pIdx}>{part.slice(1, -1)}</em>;
      }
      if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
        return (
          <code key={pIdx} style={{ backgroundColor: 'rgba(0,0,0,0.06)', padding: '0.15rem 0.35rem', borderRadius: '4px', fontFamily: 'monospace', fontSize: '0.85em' }}>
            {part.slice(1, -1)}
          </code>
        );
      }
      return part;
    });

    if (isBullet) {
      return (
        <div key={lineIdx} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.45rem', margin: '0.2rem 0 0.2rem 0.5rem' }}>
          <span style={{ color: 'var(--brand-blue)', fontWeight: 'bold' }}>•</span>
          <div>{formattedLine}</div>
        </div>
      );
    }

    return (
      <div key={lineIdx} style={{ marginBottom: '0.25rem', lineHeight: '1.5' }}>
        {formattedLine}
      </div>
    );
  });
};

export const ChatMessage: React.FC<Props> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start', width: '100%' }}>
        <div className="message-bubble">
          {renderFormattedContent(message.content)}
        </div>
      </div>
    </div>
  );
};
