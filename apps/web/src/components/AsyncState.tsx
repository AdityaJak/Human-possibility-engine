import type { ReactNode } from "react";
import { ApiError } from "../api/client";
import { apiBaseUrl } from "../api/config";

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" /> <span>{label}</span>
    </div>
  );
}

export function ErrorState({ error, onRetry, title = "Something went wrong" }: { error: Error; onRetry?: () => void; title?: string }) {
  const isApi = error instanceof ApiError;
  const network = isApi && error.isNetwork;
  return (
    <div className="state state--error" role="alert">
      <h2 className="empty-title">{network ? "Cannot reach the API" : title}</h2>
      <p>{error.message}</p>
      {network && (
        <p className="muted">
          Start the backend with <code>make api</code> (expected at <code>{apiBaseUrl()}</code>). The scientific pages only show data served by the real API.
        </p>
      )}
      {isApi && !network && (
        <p className="muted">
          HTTP {error.status} from <code>{error.url}</code>
        </p>
      )}
      {onRetry && (
        <button type="button" className="btn btn--sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="state">
      <h2 className="empty-title">{title}</h2>
      {children && <div className="muted">{children}</div>}
    </div>
  );
}

interface AsyncProps<T> {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
  reload?: () => void;
  loadingLabel?: string;
  isEmpty?: (d: T) => boolean;
  empty?: ReactNode;
  children: (d: T) => ReactNode;
}

/** Render loading / error / empty / data states for a useApi result. */
export function Async<T>({ data, error, loading, reload, loadingLabel, isEmpty, empty, children }: AsyncProps<T>) {
  if (error) return <ErrorState error={error} onRetry={reload} />;
  if (loading && data === undefined) return <Loading label={loadingLabel} />;
  if (data === undefined) return <EmptyState title="No data" />;
  if (isEmpty && isEmpty(data)) return <>{empty ?? <EmptyState title="Nothing here yet" />}</>;
  return <>{children(data)}</>;
}
