import { describe, expect, it } from "vitest";
import type { AppConfig } from "../api/types";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { redactConfigForDisplay } from "./redactConfig";

describe("redactConfigForDisplay", () => {
  it("replaces a present server.api_key with the fixed redaction token", () => {
    const config = {
      ...FALLBACK_APP_CONFIG,
      server: { api_key: "sk-super-secret-value" },
    } as unknown as AppConfig;

    const redacted = redactConfigForDisplay(config) as { server: { api_key: string } };

    expect(redacted.server.api_key).toBe("***");
    expect(JSON.stringify(redacted)).not.toContain("sk-super-secret-value");
  });

  it("leaves a null server.api_key untouched (null is not a secret)", () => {
    const config = {
      ...FALLBACK_APP_CONFIG,
      server: { api_key: null },
    } as unknown as AppConfig;

    const redacted = redactConfigForDisplay(config) as { server: { api_key: unknown } };

    expect(redacted.server.api_key).toBeNull();
  });

  it("handles a config with no server section at all (current mock/fallback shape)", () => {
    const redacted = redactConfigForDisplay(FALLBACK_APP_CONFIG) as AppConfig;

    expect(redacted.generation_presets).toEqual(FALLBACK_APP_CONFIG.generation_presets);
    expect(redacted.limits).toEqual(FALLBACK_APP_CONFIG.limits);
  });

  it("does not mutate the original config object", () => {
    const config = {
      ...FALLBACK_APP_CONFIG,
      server: { api_key: "sk-super-secret-value" },
    } as unknown as AppConfig;
    const snapshotBefore = JSON.stringify(config);

    redactConfigForDisplay(config);

    expect(JSON.stringify(config)).toBe(snapshotBefore);
    expect((config as unknown as { server: { api_key: string } }).server.api_key).toBe(
      "sk-super-secret-value",
    );
  });
});
