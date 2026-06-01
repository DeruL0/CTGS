export function computeSliceRequestSize(baseSliceSize: number, devicePixelRatio: number = 1): number {
  const base = Number.isFinite(baseSliceSize) ? baseSliceSize : 256;
  return Math.min(256, Math.max(256, Math.round(base)));
}
