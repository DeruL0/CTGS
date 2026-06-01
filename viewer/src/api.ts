import type { AxisName, GaussianBuffer, GaussianMetaPayload, SessionPayload, SliceLayer, SlicePreviewMode } from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function fetchSession(): Promise<SessionPayload> {
  const response = await fetch("/api/session");
  if (!response.ok) {
    throw new ApiError(`Failed to load session: ${response.status}`, response.status);
  }
  return response.json();
}

export async function fetchGaussianMeta(): Promise<GaussianMetaPayload> {
  const response = await fetch("/api/gaussians/meta");
  if (!response.ok) {
    throw new ApiError(`Failed to load gaussian meta: ${response.status}`, response.status);
  }
  return response.json();
}

export async function fetchGaussianBuffer(): Promise<ArrayBuffer> {
  const response = await fetch("/api/gaussians/buffer");
  if (!response.ok) {
    throw new ApiError(`Failed to load gaussian buffer: ${response.status}`, response.status);
  }
  return response.arrayBuffer();
}

export async function loadPlyFile(file: File): Promise<SessionPayload> {
  const params = new URLSearchParams({ filename: file.name });
  const response = await fetch(`/api/session/load?${params.toString()}`, {
    method: "POST",
    body: file
  });
  if (!response.ok) {
    throw new ApiError(`Failed to load PLY: ${response.status}`, response.status);
  }
  return response.json();
}

export function unpackGaussianBuffer(meta: GaussianMetaPayload, buffer: ArrayBuffer): GaussianBuffer {
  const raw = new Float32Array(buffer);
  const count = meta.count;
  const stride = meta.stride_floats;
  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const normals = new Float32Array(count * 3);
  const opacity = new Float32Array(count);
  const regionType = new Float32Array(count);
  const materialId = new Float32Array(count);
  const supportRadius = new Float32Array(count);
  const intensity = new Float32Array(count);
  const intensityField = meta.fields.attenuation ?? meta.fields.intensity;

  for (let index = 0; index < count; index += 1) {
    const base = index * stride;
    positions.set(raw.subarray(base + 0, base + 3), index * 3);
    scales.set(raw.subarray(base + 3, base + 6), index * 3);
    rotations.set(raw.subarray(base + 6, base + 10), index * 4);
    normals.set(raw.subarray(base + 10, base + 13), index * 3);
    opacity[index] = raw[base + 13];
    regionType[index] = raw[base + 14];
    materialId[index] = raw[base + 15];
    supportRadius[index] = raw[base + 16];
    intensity[index] = intensityField === undefined ? 0.0 : raw[base + intensityField.offset];
  }

  return {
    count,
    positions,
    scales,
    rotations,
    normals,
    opacity,
    regionType,
    materialId,
    supportRadius,
    intensity
  };
}

export function buildSliceUrl(
  axis: AxisName,
  t: number,
  layer: SliceLayer,
  size: number,
  options: {
    intensityClipEnabled: boolean;
    intensityMin: number;
    intensityMax: number;
    previewMode: SlicePreviewMode;
  }
): string {
  const params = new URLSearchParams({
    axis,
    t: String(t),
    layer,
    size: String(size),
    intensity_clip: options.intensityClipEnabled ? "1" : "0",
    intensity_min: String(options.intensityMin),
    intensity_max: String(options.intensityMax),
    preview: options.previewMode
  });
  return `/api/slice/gs?${params.toString()}`;
}
