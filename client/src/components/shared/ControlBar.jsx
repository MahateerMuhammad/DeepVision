export default function ControlBar({ children }) {
  return (
    <div className="bg-gray-100 p-4 border-b flex items-center justify-between">
      {children}
    </div>
  );
}
