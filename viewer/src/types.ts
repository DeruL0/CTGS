export type AxisName = "x" | "y" | "z";
export type RenderMode = "composite" | "surface-lit" | "surface-normal" | "region" | "bulk-only" | "intensity";
export type SliceLayer = "all" | "surface" | "bulk";
export type SlicePreviewMode = "raw" | "clipped" | "mask";

export interface SessionPayload {
  ply_path: string;
  gaussian_count: number;
  surface_count: number;
  bulk_count: number;
  device: string;
  bbox: {
    min: [number, number, number];
    max: [number, number, number];
    size: [number, number, number];
    center: [number, number, number];
  };
  available_axes: AxisName[];
  render_modes: RenderMode[];
  slice_layers: SliceLayer[];
  slice_preview_modes: SlicePreviewMode[];
  surface_intensity: {
    min: number;
    p01: number;
    p05: number;
    p50: number;
    p95: number;
    p99: number;
    max: number;
  };
  bulk_intensity: {
    min: number;
    p01: number;
    p05: number;
    p50: number;
    p95: number;
    p99: number;
    max: number;
  };
  defaults: {
    renderMode: RenderMode;
    axis: AxisName;
    sliceT: number;
    sliceLayer: SliceLayer;
    surfaceAlpha: number;
    bulkAlpha: number;
    sliceFadeWidthMm: number;
    clipSoftnessMm: number;
    clipEnabled: boolean;
    sliceSize: number;
    fpsLimit: number;
    splatRadiusScale: number;
    intensityClipEnabled: boolean;
    intensityMin: number;
    intensityMax: number;
    slicePreviewMode: SlicePreviewMode;
  };
}

export interface GaussianMetaPayload {
  count: number;
  stride_floats: number;
  stride_bytes: number;
  dtype: string;
  surface_count: number;
  bulk_count: number;
  fields: Record<string, { offset: number; size: number }>;
}

export interface GaussianBuffer {
  count: number;
  positions: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
  normals: Float32Array;
  opacity: Float32Array;
  regionType: Float32Array;
  materialId: Float32Array;
  supportRadius: Float32Array;
  intensity: Float32Array;
}
