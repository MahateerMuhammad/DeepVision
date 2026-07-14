import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import TopBar from "./components/layout/TopBar";
import NavRail from "./components/layout/NavRail";
import ErrorBoundary from "./components/ui/ErrorBoundary";
import { EngineProvider } from "./lib/useEngine";
import { ToastProvider } from "./components/ui/Toast";
import NetworkCanvas from "./pages/NetworkCanvas";
import ActivationLab from "./pages/ActivationLab";
import OptimizerArena from "./pages/OptimizerArena";
import CnnLab from "./pages/CnnLab";
import BatchNormLab from "./pages/BatchNormLab";

// Route content guarded by an ErrorBoundary: any render throw on a page shows a
// recoverable fallback instead of white-screening the whole app, and navigating
// to another route (pathname change) auto-clears it.
function RoutedContent() {
  const location = useLocation();
  return (
    <ErrorBoundary resetKey={location.pathname}>
      <Routes>
        <Route path="/" element={<NetworkCanvas />} />
        <Route path="/activations" element={<ActivationLab />} />
        <Route path="/optimizers" element={<OptimizerArena />} />
        <Route path="/cnn" element={<CnnLab />} />
        <Route path="/batchnorm" element={<BatchNormLab />} />
      </Routes>
    </ErrorBoundary>
  );
}

export default function App() {
  return (
    <EngineProvider>
      <ToastProvider>
        <BrowserRouter>
          <div className="flex h-full flex-col">
            <TopBar />
            <div className="flex min-h-0 flex-1 max-lg:flex-col-reverse lg:flex-row">
              <NavRail />
              <RoutedContent />
            </div>
          </div>
        </BrowserRouter>
      </ToastProvider>
    </EngineProvider>
  );
}
