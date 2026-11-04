import { createContext, useContext } from "react";
import type { FutureState } from "./model";

export interface Ctx {
  state: FutureState;
  update: (patch: Partial<FutureState> | ((s: FutureState) => Partial<FutureState>)) => void;
  reset: () => void;
}

export const FutureCtx = createContext<Ctx | null>(null);

export function useFuture(): Ctx {
  const c = useContext(FutureCtx);
  if (!c) throw new Error("useFuture must be used inside FutureProvider");
  return c;
}
