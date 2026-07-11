import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import ActivationFunctions from './pages/ActivationFunctions';
// Import other pages as they are built...

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/activation" element={<ActivationFunctions />} />
      </Routes>
    </Router>
  );
}

export default App;
