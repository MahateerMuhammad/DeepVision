import { NavLink } from "react-router-dom";

const S = { fill: "none", stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "square" };

const ICONS = {
  network: (
    <svg width="20" height="20" viewBox="0 0 20 20" {...S}>
      <circle cx="4" cy="10" r="2.2" />
      <circle cx="16" cy="5" r="2.2" />
      <circle cx="16" cy="15" r="2.2" />
      <path d="M6 9.2 13.9 5.8 M6 10.8 13.9 14.2" />
    </svg>
  ),
  activation: (
    <svg width="20" height="20" viewBox="0 0 20 20" {...S}>
      <path d="M2 15 H8 C11 15 12 5 15 5 H18" />
      <path d="M2 18 H18 M2 2 V18" strokeOpacity="0.35" />
    </svg>
  ),
  optimizer: (
    <svg width="20" height="20" viewBox="0 0 20 20" {...S}>
      <ellipse cx="10" cy="10" rx="7.5" ry="6" />
      <ellipse cx="10" cy="10" rx="4.5" ry="3.4" />
      <circle cx="10" cy="10" r="1" fill="currentColor" stroke="none" />
    </svg>
  ),
  cnn: (
    <svg width="20" height="20" viewBox="0 0 20 20" {...S}>
      <rect x="3" y="3" width="10" height="10" />
      <rect x="7" y="7" width="10" height="10" />
    </svg>
  ),
  batchnorm: (
    <svg width="20" height="20" viewBox="0 0 20 20" {...S}>
      <path d="M2 16 V11 M6 16 V6 M10 16 V3 M14 16 V8 M18 16 V12" />
      <path d="M2 16 H18" strokeOpacity="0.35" />
    </svg>
  ),
};

const ROUTES = [
  { to: "/", icon: "network", label: "Network Canvas" },
  { to: "/activations", icon: "activation", label: "Activation Lab" },
  { to: "/optimizers", icon: "optimizer", label: "Optimizer Arena" },
  { to: "/cnn", icon: "cnn", label: "CNN Lab" },
  { to: "/batchnorm", icon: "batchnorm", label: "BatchNorm" },
];

export default function NavRail() {
  return (
    <nav
      className="z-30 flex shrink-0 border-line bg-panel max-lg:h-14 max-lg:w-full max-lg:flex-row max-lg:items-center max-lg:justify-around max-lg:border-t lg:w-16 lg:flex-col lg:items-center lg:gap-1 lg:border-r lg:pt-3"
      aria-label="Modules"
    >
      {ROUTES.map((r) => (
        <NavLink
          key={r.to}
          to={r.to}
          title={r.label}
          className={({ isActive }) =>
            `group relative flex h-11 w-11 items-center justify-center transition-colors duration-150 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-cerulean ${
              isActive ? "text-ink" : "text-ink-soft hover:bg-black/[0.04] hover:text-ink"
            }`
          }
          style={{ borderRadius: 8 }}
        >
          {({ isActive }) => (
            <>
              {isActive && (
                <span className="absolute bg-ink max-lg:top-0 max-lg:left-1/2 max-lg:h-0.5 max-lg:w-6 max-lg:-translate-x-1/2 lg:top-1/2 lg:left-0 lg:h-6 lg:w-0.5 lg:-translate-y-1/2" />
              )}
              {ICONS[r.icon]}
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}
