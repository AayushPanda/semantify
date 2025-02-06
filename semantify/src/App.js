import './App.css';
import UploadButton from './Components/UploadButton';
import DownloadButton from './Components/DownloadButton';
import Header from './Components/Header';

function App() {
  return (
    <div className="App">
      <Header/>
      <UploadButton/>
      <DownloadButton/>

    </div>
  );
}

export default App;
