import { describe, expect, it } from "vitest";
import { routeSourceByExtension } from "./sourceRouting";

describe("routeSourceByExtension", () => {
  it.each(["photo.png", "photo.jpg", "photo.jpeg", "photo.webp"])("routes %s to 'image'", (fileName) => {
    expect(routeSourceByExtension(fileName)).toBe("image");
  });

  it.each(["clip.mp4", "clip.mov", "clip.webm", "clip.mkv"])("routes %s to 'video'", (fileName) => {
    expect(routeSourceByExtension(fileName)).toBe("video");
  });

  it("is case-insensitive", () => {
    expect(routeSourceByExtension("PHOTO.PNG")).toBe("image");
    expect(routeSourceByExtension("Clip.MP4")).toBe("video");
    expect(routeSourceByExtension("Mixed.WebP")).toBe("image");
    expect(routeSourceByExtension("Mixed.MKV")).toBe("video");
  });

  it("returns null for an unrecognised extension", () => {
    expect(routeSourceByExtension("document.txt")).toBeNull();
    expect(routeSourceByExtension("audio.wav")).toBeNull();
    expect(routeSourceByExtension("archive.zip")).toBeNull();
  });

  it("returns null when the file name has no extension", () => {
    expect(routeSourceByExtension("no_extension")).toBeNull();
  });

  it("returns null when the file name ends with a bare dot", () => {
    expect(routeSourceByExtension("trailing.")).toBeNull();
  });
});
