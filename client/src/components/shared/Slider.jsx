export default function Slider({ label, min, max, value, onChange, step = 0.1 }) {
  return (
    <div className="flex flex-col space-y-2">
      <label className="text-sm font-medium text-gray-700">{label}: {value}</label>
      <input 
        type="range" 
        min={min} 
        max={max} 
        value={value} 
        onChange={onChange} 
        step={step}
        className="w-full"
      />
    </div>
  );
}
