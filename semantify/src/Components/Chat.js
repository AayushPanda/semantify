import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowUp } from '@fortawesome/free-solid-svg-icons';
import axios from 'axios'; 

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      const userMessage = { text: inputValue, type: 'user' };
      setMessages([...messages, userMessage]);
      setInputValue('');

      try {
        // Send the message to the FastAPI server
        const response = await axios.post('http://localhost:8000/generate', {
          prompt: inputValue,
        });

        // Handle the response from the server
        const deepSeekMessage = { text: response.data.response, type: 'assistant' };
        setMessages((prevMessages) => [...prevMessages, deepSeekMessage]);

        // CHANGED THIS PART, links each file right now
        const sourcesMessage = { 
          text: `Sources Used:\n${response.data.files.map(file => `<a href="${file}" style="color: blue;">${file}</a>`).join('<br />')}`, 
          type: 'assistant' 
        };
        setMessages((prevMessages) => [...prevMessages, sourcesMessage]);
      } catch (error) {
        console.error('Error communicating with the server:', error);
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: 'Error communicating with the server', type: 'error' },
        ]);
      }
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.type}`} dangerouslySetInnerHTML={{ __html: message.text }} /> // Use dangerouslySetInnerHTML to render HTML
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
          <FontAwesomeIcon className="send-button-arrow" icon={faArrowUp} />
        </button>
      </form>
    </div>
  );
}