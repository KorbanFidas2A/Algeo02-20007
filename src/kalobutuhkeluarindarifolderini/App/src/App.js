import React, {useState, useEffect} from 'react';
import logo from './logo.svg';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import  Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Main from './components/Main';

function App() {
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    fetch('/time').then(res => res.json()).then(data => {setCurrentTime(data.time)});
  }, [])
  return (
    <div className="App">
    
  <Form.Group controlId="formFile" className="mb-3">
    <Form.Label>Input gambar yang akan dikompresi disini</Form.Label>
    <Form.Control type="file" />
  </Form.Group>
  <Form.Control size="sm" type="text" placeholder="Persentase kompresi gambar" />
  <Main />
      <Button variant="outline-primary">Download</Button>{' '}
    </div>
  );
}

export default App;
