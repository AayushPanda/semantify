import './App.css';
import UploadButton from './Components/UploadButton';
import DownloadButton from './Components/DownloadButton';
import Header from './Components/Header';
import Graph from './Components/Graph';
import Chat from './Components/Chat';

function App() {
  return (
    <div className="App">
      <Header />
      <div className="main-content">
        <div className="left-section">
          <Graph />
          <div className="button-container">
            <UploadButton />
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
