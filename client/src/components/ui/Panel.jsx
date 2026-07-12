export default function Panel({ title, right, className = "", bodyClassName = "", children }) {
  return (
    <section
      className={`border border-line bg-panel shadow-instrument ${className}`}
      style={{ borderRadius: 4 }}
    >
      {(title || right) && (
        <header className="flex h-8 items-center justify-between border-b border-line px-3">
          <h3 className="micro-label">{title}</h3>
          {right}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}
