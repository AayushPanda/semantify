import './App.css';
import UploadButton from './Components/UploadButton';
import DownloadButton from './Components/DownloadButton';
import Header from './Components/Header';
import Graph from './Components/Graph';
import Chat from './Components/Chat';
import React, { useRef, useEffect, useState } from 'react';
import stock_data from './data_stock.json';

function App() {
  window.localStorage.setItem("data.json", JSON.stringify(stock_data));
  const [dataFile, setDataFile] = useState(() => JSON.parse(localStorage.getItem("data.json"))); // Default file
  return (
    <div className="App">
      <Header />
      <div className="main-content">
        <div className="left-section">
          <Graph dataFile={dataFile} setDataFile={setDataFile} />
          <div className="button-container">
            <UploadButton setDataFile={setDataFile} />
            <DownloadButton />
          </div>
        </div>
        <div className="right-section">
          <Chat />
        </div>
      </div>
    </div>
  );
}

export default App;
