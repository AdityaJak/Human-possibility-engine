import type { ClaimRecord, ExperimentResult, JobStatus } from "../types/contracts";

type Tone = "ok" | "warn" | "bad" | "neutral" | "lab";

const CLAIM: Record<ClaimRecord["status"], { label: string; tone: Tone }> = {
  supported: { label: "Supported", tone: "ok" },
  not_supported: { label: "Not supported", tone: "bad" },
  inconclusive: { label: "Inconclusive", tone: "warn" },
  not_tested: { label: "Not tested", tone: "neutral" },
  blocked: { label: "Blocked", tone: "neutral" },
};

const RESULT: Record<ExperimentResult["status"], { label: string; tone: Tone }> = {
  passed: { label: "Passed", tone: "ok" },
  failed: { label: "Failed", tone: "bad" },
  inconclusive: { label: "Inconclusive", tone: "warn" },
  error: { label: "Error", tone: "bad" },
};

const JOB: Record<JobStatus, { label: string; tone: Tone }> = {
  queued: { label: "Queued", tone: "neutral" },
  running: { label: "Running", tone: "lab" },
  paused: { label: "Paused", tone: "warn" },
  completed: { label: "Completed", tone: "ok" },
  failed: { label: "Failed", tone: "bad" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};

function Chip({ label, tone, title }: { label: string; tone: Tone; title?: string }) {
  return (
    <span className={`chip chip--${tone}`} title={title}>
      {label}
    </span>
  );
}

export function ClaimStatusChip({ status }: { status: ClaimRecord["status"] }) {
  const m = CLAIM[status] ?? { label: status, tone: "neutral" as Tone };
  return <Chip {...m} />;
}

export function ResultStatusChip({ status }: { status: ExperimentResult["status"] }) {
  const m = RESULT[status] ?? { label: status, tone: "neutral" as Tone };
  return <Chip {...m} />;
}

export function JobStatusChip({ status }: { status: JobStatus }) {
  const m = JOB[status] ?? { label: status, tone: "neutral" as Tone };
  return <Chip {...m} />;
}
