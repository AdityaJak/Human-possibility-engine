import type { SafetyNotice as SafetyNoticeT } from "../types/contracts";

const TITLES: Record<SafetyNoticeT["category"], string> = {
  crisis: "If you are in danger or thinking about harming yourself",
  health: "This touches on a health decision",
  financial: "This touches on a high-stakes financial decision",
  legal: "This touches on a legal question",
  relationship: "This touches on a relationship decision",
  none: "Note",
};

export function SafetyNotice({ notice }: { notice: SafetyNoticeT | null | undefined }) {
  if (!notice || notice.category === "none") return null;
  const crisis = notice.category === "crisis";
  return (
    <section className={`notice ${crisis ? "notice--bad" : ""}`} role="alert" aria-live="assertive" data-testid="safety-notice" style={{ marginBottom: 16 }}>
      <h3 style={{ marginBottom: 6 }}>{TITLES[notice.category]}</h3>
      <p>{notice.message}</p>
      {notice.referral && (
        <p>
          <strong>{notice.referral}</strong>
        </p>
      )}
      <p className="muted" style={{ fontSize: "0.85rem" }}>
        Future Self is an educational exploration tool. It does not give medical, legal, financial or relationship verdicts.
      </p>
    </section>
  );
}
