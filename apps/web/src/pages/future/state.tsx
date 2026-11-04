import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { DEFAULT_VALUES, emptyDraft } from "./model";
import type { FutureState } from "./model";
import { FutureCtx } from "./context";
import type { Ctx } from "./context";
import { newDemoUserId, readLocal, removeLocal, writeLocal } from "../../lib/storage";

const KEY = "hpe.futureSelf.v1";

function freshState(): FutureState {
  return {
    userId: newDemoUserId(),
    consent: { demo: false, reflections: false, personalization: false },
    values: DEFAULT_VALUES.map((v) => ({ ...v })),
    draft: emptyDraft(),
    response: null,
    plan: [],
  };
}

function load(): FutureState {
  const saved = readLocal<FutureState | null>(KEY, null);
  // Only state saved after consent is ever written, so a saved state implies prior consent.
  if (saved && saved.userId && saved.consent?.demo) return { ...freshState(), ...saved };
  return freshState();
}

export function FutureProvider({ children, initial }: { children: ReactNode; initial?: Partial<FutureState> }) {
  const [state, setState] = useState<FutureState>(() => ({ ...load(), ...initial }));

  useEffect(() => {
    // Persist locally only once the user has consented; never before.
    if (state.consent.demo) writeLocal(KEY, state);
  }, [state]);

  const update = useCallback<Ctx["update"]>((patch) => {
    setState((s) => ({ ...s, ...(typeof patch === "function" ? patch(s) : patch) }));
  }, []);

  const reset = useCallback(() => {
    removeLocal(KEY);
    setState(freshState());
  }, []);

  const value = useMemo(() => ({ state, update, reset }), [state, update, reset]);
  return <FutureCtx.Provider value={value}>{children}</FutureCtx.Provider>;
}

