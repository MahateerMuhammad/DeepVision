import { BrowserRouter, Routes, Route } from "react-router-dom";
import TopBar from "./components/layout/TopBar";
import NavRail from "./components/layout/NavRail";
import { EngineProvider } from "./lib/useEngine";
import { ToastProvider } from "./components/ui/Toast";
import NetworkCanvas from "./pages/NetworkCanvas";
import ActivationLab from "./pages/ActivationLab";
import OptimizerArena from "./pages/OptimizerArena";
import CnnLab from "./pages/CnnLab";

export default function App() {
  return (
    <EngineProvider>
      <ToastProvider>
        <BrowserRouter>
          <div className="flex h-full flex-col">
            <TopBar />
            <div className="flex min-h-0 flex-1 max-lg:flex-col-reverse lg:flex-row">
              <NavRail />
              <Routes>
                <Route path="/" element={<NetworkCanvas />} />
                <Route path="/activations" element={<ActivationLab />} />
                <Route path="/optimizers" element={<OptimizerArena />} />
                <Route path="/cnn" element={<CnnLab />} />
              </Routes>
            </div>
          </div>
        </BrowserRouter>
      </ToastProvider>
    </EngineProvider>
  );
}
