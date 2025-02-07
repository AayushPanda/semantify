import React, { useState } from 'react';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      setMessages([...messages, { text: inputValue, type: 'user' }]);
      setInputValue('');
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.type}`}>
            {message.text}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="chat-input-container">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type your question here"
          className="chat-input"
        />
        <button type="submit" className="send-button">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 20V4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <path d="M5 12L12 4L19 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </form>
    </div>
  );
}