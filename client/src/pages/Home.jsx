import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="p-8">
      <h1 className="text-4xl font-bold mb-6">Deep Learning Visualizer</h1>
      <ul className="space-y-4">
        <li><Link to="/activation" className="text-blue-500 hover:underline">1. Activation Functions</Link></li>
        <li><Link to="/gradient-descent" className="text-blue-500 hover:underline">2. Gradient Descent</Link></li>
        <li><Link to="/forward-prop" className="text-blue-500 hover:underline">3. Forward Propagation</Link></li>
        <li><Link to="/backprop" className="text-blue-500 hover:underline">4. Backpropagation</Link></li>
        <li><Link to="/cnn" className="text-blue-500 hover:underline">5. CNN Filters</Link></li>
        <li><Link to="/attention" className="text-blue-500 hover:underline">6. Attention</Link></li>
        <li><Link to="/transformers" className="text-blue-500 hover:underline">7. Transformers</Link></li>
      </ul>
    </div>
  );
}
