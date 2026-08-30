import React from 'react';
import { type ChatMessage as ChatMessageType } from '../../api';

interface Props {
  message: ChatMessageType;
  isStreaming?: boolean;
}



export const ChatMessage: React.FC<Props> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start', width: '100%' }}>
        
        <div className="message-bubble">
          {message.content}
        </div>
      </div>
    </div>
  );
};
