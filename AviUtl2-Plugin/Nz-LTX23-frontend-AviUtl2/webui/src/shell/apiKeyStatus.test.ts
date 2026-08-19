import { describe, expect, it } from "vitest";
import type { AppConfig } from "../api/types";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { apiKeyStatus } from "./apiKeyStatus";

describe("apiKeyStatus", () => {
  it("returns 'set' for a non-empty string api_key", () => {
    const config = { ...FALLBACK_APP_CONFIG, server: { api_key: "sk-super-secret-value" } } as unknown as AppConfig;

    expect(apiKeyStatus(config, false)).toBe("set");
  });

  it("returns 'unset' for a null api_key", () => {
    const config = { ...FALLBACK_APP_CONFIG, server: { api_key: null } } as unknown as AppConfig;

    expect(apiKeyStatus(config, false)).toBe("unset");
  });

  it("returns 'unset' for an empty-string api_key", () => {
    const config = { ...FALLBACK_APP_CONFIG, server: { api_key: "" } } as unknown as AppConfig;

    expect(apiKeyStatus(config, false)).toBe("unset");
  });

  it("returns 'unset' when the server section is entirely absent", () => {
    expect(apiKeyStatus(FALLBACK_APP_CONFIG, false)).toBe("unset");
  });

  it("returns 'unknown' when usingFallback is true, regardless of api_key", () => {
    const config = { ...FALLBACK_APP_CONFIG, server: { api_key: "sk-super-secret-value" } } as unknown as AppConfig;

    expect(apiKeyStatus(config, true)).toBe("unknown");
  });

  it("usingFallback takes priority even over a present, non-empty api_key", () => {
    const config = { ...FALLBACK_APP_CONFIG, server: { api_key: "sk-still-present" } } as unknown as AppConfig;

    expect(apiKeyStatus(config, true)).not.toBe("set");
    expect(apiKeyStatus(config, true)).toBe("unknown");
  });
});
