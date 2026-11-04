import { lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Landing } from "./pages/Landing";

const LiveSociety = lazy(() => import("./pages/lab/LiveSociety"));
const AgentInspector = lazy(() => import("./pages/lab/AgentInspector"));
const ProofOfLearning = lazy(() => import("./pages/lab/ProofOfLearning"));
const TwinExperiment = lazy(() => import("./pages/lab/TwinExperiment"));
const EvidencePage = lazy(() => import("./pages/lab/EvidencePage"));
const Benchmarks = lazy(() => import("./pages/lab/Benchmarks"));
const FutureSelf = lazy(() => import("./pages/future/FutureSelf"));
const InvestorDemo = lazy(() => import("./pages/demo/InvestorDemo"));
const NotFound = lazy(() => import("./pages/NotFound"));

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Landing />} />
        <Route path="lab" element={<Navigate to="/lab/society" replace />} />
        <Route path="lab/society" element={<LiveSociety />} />
        <Route path="lab/runs/:runId/agents/:agentId" element={<AgentInspector />} />
        <Route path="lab/learning" element={<ProofOfLearning />} />
        <Route path="lab/experiment" element={<TwinExperiment />} />
        <Route path="lab/evidence" element={<EvidencePage />} />
        <Route path="lab/benchmarks" element={<Benchmarks />} />
        <Route path="future" element={<Navigate to="/future/welcome" replace />} />
        <Route path="future/:step" element={<FutureSelf />} />
        <Route path="demo" element={<InvestorDemo />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
