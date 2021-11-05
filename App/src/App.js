import React, {useState, useEffect} from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    fetch('/time').then(res => res.json()).then(data => {setCurrentTime(data.time)});
  }, [])
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Selamat datang!
        </p>
        <a
          className="App-link"
          href="https://github.com/KorbanFidas2A/Algeo02-20007.git"
          target="_blank"
          rel="noopener noreferrer"
        >
          Image Compression menggunakan SVD
        </a>
        <p>The current time is {currentTime}.</p>
      </header>
    </div>
  );
}

export default App;
