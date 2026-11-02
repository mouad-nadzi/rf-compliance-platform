import React from 'react';
import { type ChatMessage as ChatMessageType } from '../../api';

interface Props {
  message: ChatMessageType;
  isStreaming?: boolean;
}

interface Block {
  type: 'heading' | 'table' | 'code' | 'blockquote' | 'bullet_list' | 'numbered_list' | 'hr' | 'paragraph';
  level?: number;
  text?: string;
  lang?: string;
  code?: string;
  tableHeaders?: string[];
  tableRows?: string[][];
  items?: string[];
}

/**
 * Parses inline formatting tokens: links, inline code, bold-italic, bold, italic, and URLs.
 */
const renderInline = (text: string): React.ReactNode[] => {
  if (!text) return [];

  // Match: [title](url), `code`, ***bold-italic***, **bold**, *italic*, https?://...
  const tokenRegex = /(\[.*?\]\(https?:\/\/[^\s\)]+\)|`.*?`|\*\*\*.*?\*\*\*|\*\*.*?\*\*|\*.*?\*|https?:\/\/[^\s\)]+)/g;
  const parts = text.split(tokenRegex);

  return parts.map((part, pIdx) => {
    if (!part) return null;

    // Link: [title](url)
    const linkMatch = part.match(/^\[(.*?)\]\((https?:\/\/[^\s\)]+)\)$/);
    if (linkMatch) {
      return (
        <a
          key={pIdx}
          href={linkMatch[2]}
          target="_blank"
          rel="noreferrer"
          style={{
            color: 'var(--msg-accent-color, var(--brand-blue))',
            textDecoration: 'underline',
            textUnderlineOffset: '3px',
            fontWeight: 500,
            wordBreak: 'break-all'
          }}
        >
          {linkMatch[1]}
        </a>
      );
    }

    // Inline Code: `code`
    if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
      return (
        <code
          key={pIdx}
          style={{
            backgroundColor: 'var(--msg-code-bg, rgba(0, 0, 0, 0.06))',
            padding: '0.15rem 0.35rem',
            borderRadius: '4px',
            fontFamily: 'monospace',
            fontSize: '0.88em',
            color: 'var(--msg-text-color, var(--text-primary))'
          }}
        >
          {part.slice(1, -1)}
        </code>
      );
    }

    // Bold Italic: ***text***
    if (part.startsWith('***') && part.endsWith('***') && part.length >= 6) {
      return (
        <strong key={pIdx} style={{ fontWeight: 700, fontStyle: 'italic', color: 'var(--msg-text-color, var(--text-primary))' }}>
          {part.slice(3, -3)}
        </strong>
      );
    }

    // Bold: **text**
    if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
      return (
        <strong key={pIdx} style={{ fontWeight: 700, color: 'var(--msg-text-color, var(--text-primary))' }}>
          {part.slice(2, -2)}
        </strong>
      );
    }

    // Italic: *text*
    if (part.startsWith('*') && part.endsWith('*') && part.length >= 2) {
      return <em key={pIdx} style={{ color: 'inherit' }}>{part.slice(1, -1)}</em>;
    }

    // Raw URL: https://...
    if (/^https?:\/\/[^\s\)]+$/.test(part)) {
      return (
        <a
          key={pIdx}
          href={part}
          target="_blank"
          rel="noreferrer"
          style={{
            color: 'var(--msg-accent-color, var(--brand-blue))',
            textDecoration: 'underline',
            textUnderlineOffset: '3px',
            fontWeight: 500,
            wordBreak: 'break-all'
          }}
        >
          {part}
        </a>
      );
    }

    return part;
  });
};

/**
 * Tokenizes markdown content into structured block-level elements (headings, tables, lists, quotes, paragraphs).
 */
const parseMarkdownBlocks = (content: string): Block[] => {
  const lines = content.split(/\r?\n/);
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i++;
      continue;
    }

    // 1. Fenced Code Block: ```lang
    if (trimmed.startsWith('```')) {
      const lang = trimmed.slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      if (i < lines.length) i++;
      blocks.push({
        type: 'code',
        lang,
        code: codeLines.join('\n')
      });
      continue;
    }

    // 2. Headings: #, ##, ###, ####, #####, ######
    const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      blocks.push({
        type: 'heading',
        level: headingMatch[1].length,
        text: headingMatch[2].trim()
      });
      i++;
      continue;
    }

    // 3. Horizontal Rule: ---, ***, ___
    if (/^(\-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      blocks.push({ type: 'hr' });
      i++;
      continue;
    }

    // 4. Blockquote: > text
    if (trimmed.startsWith('>')) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ''));
        i++;
      }
      blocks.push({
        type: 'blockquote',
        text: quoteLines.join('\n')
      });
      continue;
    }

    // 5. Table: | col1 | col2 | ... followed by |---|---|
    if (
      trimmed.startsWith('|') &&
      trimmed.endsWith('|') &&
      i + 1 < lines.length &&
      /^\|(\s*:?-+:?\s*\|)+$/.test(lines[i + 1].trim())
    ) {
      const headerRow = trimmed
        .slice(1, -1)
        .split('|')
        .map(c => c.trim());

      i += 2; // skip header and delimiter line

      const tableRows: string[][] = [];
      while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
        const rowCells = lines[i]
          .trim()
          .slice(1, -1)
          .split('|')
          .map(c => c.trim());
        tableRows.push(rowCells);
        i++;
      }

      blocks.push({
        type: 'table',
        tableHeaders: headerRow,
        tableRows
      });
      continue;
    }

    // 6. Bullet List: - item, * item, + item, • item
    if (/^[\-\*\+•]\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^[\-\*\+•]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[\-\*\+•]\s+/, ''));
        i++;
      }
      blocks.push({
        type: 'bullet_list',
        items
      });
      continue;
    }

    // 7. Numbered List: 1. item, 2. item
    if (/^\d+\.\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
        i++;
      }
      blocks.push({
        type: 'numbered_list',
        items
      });
      continue;
    }

    // 8. Regular Paragraph
    const paraLines: string[] = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith('```') &&
      !lines[i].trim().match(/^#{1,6}\s+/) &&
      !lines[i].trim().startsWith('>') &&
      !/^(\-{3,}|\*{3,}|_{3,})$/.test(lines[i].trim()) &&
      !/^[\-\*\+•]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim()) &&
      !(lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|'))
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    blocks.push({
      type: 'paragraph',
      text: paraLines.join('\n')
    });
  }

  return blocks;
};

const renderFormattedContent = (content: string) => {
  if (!content) return null;

  const blocks = parseMarkdownBlocks(content);

  return blocks.map((block, idx) => {
    switch (block.type) {
      case 'heading': {
        const textNodes = renderInline(block.text || '');
        if (block.level === 1) {
          return (
            <h1 key={idx} style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--msg-text-color, var(--text-primary))', margin: '0.85rem 0 0.4rem 0' }}>
              {textNodes}
            </h1>
          );
        }
        if (block.level === 2) {
          return (
            <h2 key={idx} style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--msg-text-color, var(--text-primary))', margin: '0.75rem 0 0.35rem 0' }}>
              {textNodes}
            </h2>
          );
        }
        if (block.level === 3) {
          return (
            <h3 key={idx} style={{ fontSize: '1.02rem', fontWeight: 600, color: 'var(--msg-text-color, var(--text-primary))', margin: '0.65rem 0 0.3rem 0' }}>
              {textNodes}
            </h3>
          );
        }
        return (
          <h4 key={idx} style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--msg-text-color, var(--text-primary))', margin: '0.55rem 0 0.25rem 0' }}>
            {textNodes}
          </h4>
        );
      }

      case 'table': {
        return (
          <div key={idx} className="chat-table-wrapper">
            <table className="chat-table">
              <thead>
                <tr>
                  {block.tableHeaders?.map((th, hIdx) => (
                    <th key={hIdx}>
                      {renderInline(th)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.tableRows?.map((row, rIdx) => (
                  <tr key={rIdx}>
                    {row.map((cell, cIdx) => {
                      const trimmed = cell.trim();
                      
                      // Check for date pattern YYYY-MM-DD
                      const isDatePattern = /^\d{4}-\d{2}-\d{2}$/.test(trimmed);
                      if (isDatePattern) {
                        const year = parseInt(trimmed.slice(0, 4), 10);
                        if (year >= 2900) {
                          return (
                            <td key={cIdx}>
                              <span className="chat-table-badge" style={{ background: 'rgba(16, 185, 129, 0.12)', color: '#059669', fontWeight: 600 }}>
                                Infinite
                              </span>
                            </td>
                          );
                        }
                        const isExpired = year < 2026 || new Date(trimmed) < new Date();
                        return (
                          <td key={cIdx}>
                            <span className={isExpired ? 'chat-table-date-expired' : 'chat-table-badge'}>
                              {trimmed}
                            </span>
                          </td>
                        );
                      }

                      // Check for status strings
                      const upper = trimmed.toUpperCase();
                      if (['INDEFINITE', 'PERPETUAL', 'INFINITE', 'PERMANENT'].includes(upper)) {
                        return (
                          <td key={cIdx}>
                            <span className="chat-table-badge" style={{ background: 'rgba(16, 185, 129, 0.12)', color: '#059669', fontWeight: 600 }}>
                              Infinite
                            </span>
                          </td>
                        );
                      }
                      if (['EXPIRED', 'INVALID', 'REJECTED'].includes(upper)) {
                        return (
                          <td key={cIdx}>
                            <span className="chat-table-date-expired">{trimmed}</span>
                          </td>
                        );
                      }
                      if (['VALID', 'ACTIVE', 'APPROVED', 'COMPLIANT'].includes(upper)) {
                        return (
                          <td key={cIdx}>
                            <span className="chat-table-badge">{trimmed}</span>
                          </td>
                        );
                      }

                      return (
                        <td key={cIdx}>
                          {renderInline(cell)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      }

      case 'bullet_list': {
        return (
          <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', margin: '0.4rem 0 0.6rem 0' }}>
            {block.items?.map((item, itemIdx) => (
              <div key={itemIdx} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', marginLeft: '0.4rem' }}>
                <span style={{ color: 'var(--msg-accent-color, var(--brand-blue))', fontWeight: 700, lineHeight: '1.4' }}>•</span>
                <div style={{ flex: 1, lineHeight: '1.55' }}>{renderInline(item)}</div>
              </div>
            ))}
          </div>
        );
      }

      case 'numbered_list': {
        return (
          <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', margin: '0.4rem 0 0.6rem 0' }}>
            {block.items?.map((item, itemIdx) => (
              <div key={itemIdx} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', marginLeft: '0.4rem' }}>
                <span style={{ color: 'var(--msg-accent-color, var(--brand-blue))', fontWeight: 600, fontSize: '0.85rem', lineHeight: '1.6', minWidth: '1.2rem' }}>
                  {itemIdx + 1}.
                </span>
                <div style={{ flex: 1, lineHeight: '1.55' }}>{renderInline(item)}</div>
              </div>
            ))}
          </div>
        );
      }

      case 'blockquote': {
        return (
          <div
            key={idx}
            style={{
              borderLeft: '3.5px solid var(--msg-accent-color, var(--brand-blue))',
              padding: '0.45rem 0.85rem',
              margin: '0.6rem 0',
              background: 'var(--msg-quote-bg, rgba(36, 56, 129, 0.04))',
              borderRadius: '0 6px 6px 0',
              lineHeight: '1.55'
            }}
          >
            {renderInline(block.text || '')}
          </div>
        );
      }

      case 'code': {
        return (
          <div
            key={idx}
            style={{
              margin: '0.75rem 0',
              borderRadius: '8px',
              overflow: 'hidden',
              border: '1px solid var(--border-color)',
              background: '#1e293b',
              color: '#f8fafc'
            }}
          >
            {block.lang && (
              <div
                style={{
                  padding: '0.3rem 0.75rem',
                  background: '#0f172a',
                  fontSize: '0.75rem',
                  color: '#94a3b8',
                  textTransform: 'uppercase',
                  fontWeight: 600,
                  letterSpacing: '0.05em'
                }}
              >
                {block.lang}
              </div>
            )}
            <pre style={{ padding: '0.75rem', margin: 0, overflowX: 'auto', fontFamily: 'monospace', fontSize: '0.85rem', lineHeight: '1.5' }}>
              <code>{block.code}</code>
            </pre>
          </div>
        );
      }

      case 'hr': {
        return <hr key={idx} style={{ border: 'none', borderTop: '1px solid var(--msg-border-color, var(--border-color))', margin: '1rem 0' }} />;
      }

      case 'paragraph':
      default: {
        return (
          <div key={idx} style={{ marginBottom: '0.5rem', lineHeight: '1.6' }}>
            {renderInline(block.text || '')}
          </div>
        );
      }
    }
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
