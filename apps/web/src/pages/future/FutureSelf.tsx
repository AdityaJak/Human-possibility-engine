import { Link, Navigate, useParams } from "react-router-dom";
import { FutureBanner } from "../../components/Badges";
import { ConsentGate, NeedsScenarios } from "./ConsentGate";
import { FutureProvider } from "./state";
import { useFuture } from "./context";
import { STEPS, stepIndex } from "./model";
import type { StepId } from "./model";
import { ConsentStep, NorthStar, Welcome } from "./StepsIntro";
import { DecisionStudio, EvidenceUncertainty, PossibleFutures } from "./StepsDecision";
import { Conversation, ExperimentPlanner, Journey, Privacy } from "./StepsAfter";

function Stepper({ current }: { current: StepId }) {
  const { state } = useFuture();
  return (
    <nav aria-label="Future Self steps">
      <ol className="stepper">
        {STEPS.map((s, i) => {
          const locked = (s.needsConsent && !state.consent.demo) || (s.needsScenarios && !state.response);
          return (
            <li key={s.id}>
              <Link to={`/future/${s.id}`} className={s.id === current ? "active" : ""} aria-current={s.id === current ? "step" : undefined} aria-disabled={locked ? "true" : undefined}>
                <span className="num">{i + 1}</span> {s.label}
              </Link>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function StepBody({ step }: { step: StepId }) {
  const { state } = useFuture();
  const meta = STEPS[stepIndex(step)];
  let body: JSX.Element;
  switch (step) {
    case "welcome":
      body = <Welcome />;
      break;
    case "consent":
      body = <ConsentStep />;
      break;
    case "north-star":
      body = <NorthStar />;
      break;
    case "decision":
      body = <DecisionStudio />;
      break;
    case "futures":
      body = <PossibleFutures />;
      break;
    case "evidence":
      body = <EvidenceUncertainty />;
      break;
    case "plan":
      body = <ExperimentPlanner />;
      break;
    case "journey":
      body = <Journey />;
      break;
    case "conversation":
      body = <Conversation />;
      break;
    case "privacy":
      body = <Privacy />;
      break;
  }
  if (meta?.needsScenarios) body = <NeedsScenarios has={!!state.response}>{body}</NeedsScenarios>;
  if (meta?.needsConsent) body = <ConsentGate consented={state.consent.demo}>{body}</ConsentGate>;
  return body;
}

export function FutureSelfFlow({ step }: { step: StepId }) {
  return (
    <div className="stack" data-zone="future">
      <FutureBanner />
      <header className="page-head">
        <div className="eyebrow">Future Self · decision exploration</div>
        <h1>Meet your possible future selves</h1>
      </header>
      <Stepper current={step} />
      <StepBody step={step} />
    </div>
  );
}

export default function FutureSelf() {
  const { step = "welcome" } = useParams();
  if (stepIndex(step) < 0) return <Navigate to="/future/welcome" replace />;
  return (
    <FutureProvider>
      <FutureSelfFlow step={step as StepId} />
    </FutureProvider>
  );
}
