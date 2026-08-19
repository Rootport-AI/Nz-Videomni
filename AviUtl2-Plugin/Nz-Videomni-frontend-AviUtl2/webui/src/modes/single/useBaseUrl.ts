import { useEffect, useRef, useState } from "react";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";

/** Fetches the backend's origin once on mount, for building direct
 * `<video src>` URLs to the completed job's mp4 (Docs/API_REFERENCE.md
 * §3.16 — bypasses the bridge/CORS entirely, same-origin resource fetch). */
export function useBaseUrl(nativeBridge: NativeBridge = defaultBridge): string | null {
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    nativeBridge
      .request("backend.getBaseUrl", {})
      .then((result) => {
        if (mountedRef.current) setBaseUrl(result.baseUrl);
      })
      .catch(() => {
        // Non-fatal: the preview simply won't render until this resolves;
        // a retry isn't needed since the value never changes at runtime.
      });
  }, [nativeBridge]);

  return baseUrl;
}
