import { startTransition, useDeferredValue, useEffect, useRef, useState, type ChangeEvent } from "react";
import {
  Aperture,
  Blend,
  Box,
  BoxSelect,
  Columns3,
  Eye,
  FlipHorizontal2,
  FolderOpen,
  Gauge,
  Layers3,
  Orbit,
  Palette,
  RotateCcw,
  ScanLine,
  Scissors,
  Waves
} from "lucide-react";

import { ApiError, buildSliceUrl, fetchGaussianBuffer, fetchGaussianMeta, fetchSession, loadPlyFile, unpackGaussianBuffer } from "./api";
import { GaussianViewport } from "./components/GaussianViewport";
import { SlicePanel } from "./components/SlicePanel";
import { computeSliceRequestSize } from "./slicePreview";
import type { AxisName, GaussianBuffer, GaussianMetaPayload, RenderMode, SessionPayload, SliceLayer, SlicePreviewMode } from "./types";

const MODE_BUTTONS: Array<{ value: RenderMode; label: string; icon: typeof Layers3 }> = [
  { value: "composite", label: "Composite", icon: Layers3 },
  { value: "surface-lit", label: "Surface", icon: Aperture },
  { value: "surface-normal", label: "Normal", icon: Orbit },
  { value: "bulk-only", label: "Bulk", icon: Box },
  { value: "intensity", label: "Intensity", icon: Palette },
  { value: "region", label: "Region", icon: Blend }
];
const SLICE_REQUEST_DEBOUNCE_MS = 120;
const PREVIEW_MODES: SlicePreviewMode[] = ["raw", "clipped", "mask"];

export default function App() {
  const [session, setSession] = useState<SessionPayload | null>(null);
  const [gaussianMeta, setGaussianMeta] = useState<GaussianMetaPayload | null>(null);
  const [gaussians, setGaussians] = useState<GaussianBuffer | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [renderMode, setRenderMode] = useState<RenderMode>("composite");
  const [axis, setAxis] = useState<AxisName>("z");
  const [sliceT, setSliceT] = useState(0.5);
  const [sliceLayer, setSliceLayer] = useState<SliceLayer>("all");
  const [surfaceAlpha, setSurfaceAlpha] = useState(0.92);
  const [bulkAlpha, setBulkAlpha] = useState(0.28);
  const [sliceFadeWidthMm, setSliceFadeWidthMm] = useState(1.0);
  const [clipSoftnessMm, setClipSoftnessMm] = useState(0.25);
  const [clipEnabled, setClipEnabled] = useState(false);
  const [clipFlip, setClipFlip] = useState(false);
  const [fpsLimit, setFpsLimit] = useState(60);
  const [splatRadiusScale, setSplatRadiusScale] = useState(1.0);
  const [intensityClipEnabled, setIntensityClipEnabled] = useState(false);
  const [intensityMin, setIntensityMin] = useState(0.0);
  const [intensityMax, setIntensityMax] = useState(1.0);
  const [slicePreviewMode, setSlicePreviewMode] = useState<SlicePreviewMode>("raw");
  const [sliceImageUrl, setSliceImageUrl] = useState<string | null>(null);
  const [sliceLoading, setSliceLoading] = useState(false);
  const [fileLoading, setFileLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const deferredSliceT = useDeferredValue(sliceT);
  const deferredSliceLayer = useDeferredValue(sliceLayer);
  const deferredIntensityMin = useDeferredValue(intensityMin);
  const deferredIntensityMax = useDeferredValue(intensityMax);
  const deferredIntensityClipEnabled = useDeferredValue(intensityClipEnabled);
  const deferredSlicePreviewMode = useDeferredValue(slicePreviewMode);

  function clearSliceImage() {
    setSliceImageUrl((previous) => {
      if (previous !== null) URL.revokeObjectURL(previous);
      return null;
    });
  }

  function applySessionDefaults(sessionPayload: SessionPayload) {
    setSession(sessionPayload);
    setRenderMode(sessionPayload.defaults.renderMode);
    setAxis(sessionPayload.defaults.axis);
    setSliceT(sessionPayload.defaults.sliceT);
    setSliceLayer(sessionPayload.defaults.sliceLayer);
    setSurfaceAlpha(sessionPayload.defaults.surfaceAlpha);
    setBulkAlpha(sessionPayload.defaults.bulkAlpha);
    setSliceFadeWidthMm(sessionPayload.defaults.sliceFadeWidthMm);
    setClipSoftnessMm(sessionPayload.defaults.clipSoftnessMm);
    setClipEnabled(sessionPayload.defaults.clipEnabled);
    setClipFlip(false);
    setFpsLimit(sessionPayload.defaults.fpsLimit);
    setSplatRadiusScale(sessionPayload.defaults.splatRadiusScale);
    setIntensityClipEnabled(sessionPayload.defaults.intensityClipEnabled);
    setIntensityMin(sessionPayload.defaults.intensityMin);
    setIntensityMax(sessionPayload.defaults.intensityMax);
    setSlicePreviewMode(sessionPayload.defaults.slicePreviewMode);
  }

  async function hydrateViewer(sessionPayload: SessionPayload) {
    const [metaPayload, bufferPayload] = await Promise.all([
      fetchGaussianMeta(),
      fetchGaussianBuffer()
    ]);
    const unpacked = unpackGaussianBuffer(metaPayload, bufferPayload);
    applySessionDefaults(sessionPayload);
    setGaussianMeta(metaPayload);
    setGaussians(unpacked);
    clearSliceImage();
    setLoadError(null);
  }

  useEffect(() => {
    let cancelled = false;
    async function loadViewer() {
      try {
        const sessionPayload = await fetchSession();
        if (cancelled) return;
        await hydrateViewer(sessionPayload);
      } catch (error) {
        if (!cancelled) {
          if (error instanceof ApiError && error.status === 404) {
            setLoadError(null);
            return;
          }
          setLoadError(error instanceof Error ? error.message : "Failed to load viewer data.");
        }
      }
    }
    void loadViewer();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (session === null) return undefined;
    const deviceScale = Math.min(Math.max(Math.ceil(window.devicePixelRatio || 1), 1), 2);
    const sliceSize = computeSliceRequestSize(session.defaults.sliceSize, deviceScale);
    let cancelled = false;
    const abortController = new AbortController();
    const debounceId = window.setTimeout(() => {
      setSliceLoading(true);
      void updateSlice();
    }, SLICE_REQUEST_DEBOUNCE_MS);

    async function updateSlice() {
      try {
        const response = await fetch(buildSliceUrl(axis, deferredSliceT, deferredSliceLayer, sliceSize, {
          intensityClipEnabled: deferredIntensityClipEnabled,
          intensityMin: deferredIntensityMin,
          intensityMax: deferredIntensityMax,
          previewMode: deferredSlicePreviewMode
        }), {
          signal: abortController.signal
        });
        if (!response.ok) throw new Error(`Slice request failed: ${response.status}`);
        const blob = await response.blob();
        if (cancelled) return;
        const nextUrl = URL.createObjectURL(blob);
        setSliceImageUrl((previous) => {
          if (previous !== null) URL.revokeObjectURL(previous);
          return nextUrl;
        });
      } catch (error) {
        const isAbort = error instanceof Error && error.name === "AbortError";
        if (!cancelled && !isAbort) {
          setLoadError(error instanceof Error ? error.message : "Failed to fetch slice preview.");
        }
      } finally {
        if (!cancelled) setSliceLoading(false);
      }
    }

    return () => {
      cancelled = true;
      window.clearTimeout(debounceId);
      abortController.abort();
    };
  }, [axis, deferredIntensityClipEnabled, deferredIntensityMax, deferredIntensityMin, deferredSliceLayer, deferredSlicePreviewMode, deferredSliceT, session]);

  useEffect(() => {
    return () => {
      if (sliceImageUrl !== null) URL.revokeObjectURL(sliceImageUrl);
    };
  }, [sliceImageUrl]);

  async function handlePlyFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (file === undefined) return;
    setFileLoading(true);
    setLoadError(null);
    setSliceLoading(false);
    clearSliceImage();
    try {
      const loadedSession = await loadPlyFile(file);
      await hydrateViewer(loadedSession);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Failed to load PLY.");
    } finally {
      setFileLoading(false);
    }
  }

  const maxSliceFade = session ? Math.max(...session.bbox.size) * 0.6 : 5.0;
  const maxClipSoftness = session ? Math.max(...session.bbox.size) * 0.2 : 2.0;
  const bulkIntensityStats = session?.bulk_intensity;
  const surfaceIntensityStats = session?.surface_intensity;
  const intensityDomainMin = 0.0;
  const intensityDomainMax = 1.0;

  function clampIntensity(value: number) {
    return Math.min(intensityDomainMax, Math.max(intensityDomainMin, value));
  }

  function updateIntensityMin(value: number) {
    const nextValue = clampIntensity(value);
    setIntensityMin(Math.min(nextValue, intensityMax));
  }

  function updateIntensityMax(value: number) {
    const nextValue = clampIntensity(value);
    setIntensityMax(Math.max(nextValue, intensityMin));
  }

  function resetIntensityClip() {
    setIntensityMin(0.0);
    setIntensityMax(1.0);
    setSlicePreviewMode("raw");
    setIntensityClipEnabled(false);
  }

  return (
    <div className="app-shell">
      <header className="app-toolbar">
        <div className="toolbar-group">
          <div className="title-block">
            <span className="eyebrow">CTGS Viewer</span>
            <h1>CTGS Inspection</h1>
          </div>
          {session !== null ? (
            <div className="session-statline">
              <span>{session.gaussian_count.toLocaleString()} G</span>
              <span>{session.surface_count.toLocaleString()} surf</span>
              <span>{session.bulk_count.toLocaleString()} bulk</span>
              <span>{session.device}</span>
            </div>
          ) : null}
          <input
            ref={fileInputRef}
            accept=".ply"
            className="hidden-file-input"
            onChange={handlePlyFileChange}
            type="file"
          />
          <button
            className={`icon-toggle compact ${fileLoading ? "is-active" : ""}`}
            disabled={fileLoading}
            onClick={() => fileInputRef.current?.click()}
            title="Open PLY"
            type="button"
          >
            <FolderOpen size={16} />
            <span>{fileLoading ? "Loading…" : "Open PLY"}</span>
          </button>
        </div>

        <div className="toolbar-group toolbar-modes">
          {MODE_BUTTONS.map((button) => {
            const Icon = button.icon;
            return (
              <button
                key={button.value}
                className={`icon-toggle ${renderMode === button.value ? "is-active" : ""}`}
                title={button.label}
                onClick={() => setRenderMode(button.value)}
                type="button"
              >
                <Icon size={16} />
                <span>{button.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      <main className="workspace">
        <section className="viewport-column">
          <div className="control-strip">
            <div className="segmented-group" aria-label="Slice axis">
              {(["x", "y", "z"] as AxisName[]).map((candidate) => (
                <button
                  key={candidate}
                  className={axis === candidate ? "is-active" : ""}
                  onClick={() => setAxis(candidate)}
                  type="button"
                >
                  {candidate.toUpperCase()}
                </button>
              ))}
            </div>

            <div className="segmented-group" aria-label="Slice layer">
              {(["all", "surface", "bulk"] as SliceLayer[]).map((candidate) => (
                <button
                  key={candidate}
                  className={sliceLayer === candidate ? "is-active" : ""}
                  onClick={() => setSliceLayer(candidate)}
                  type="button"
                >
                  {candidate}
                </button>
              ))}
            </div>

            <div className="button-cluster" aria-label="Clip plane">
              <button
                className={`icon-toggle compact ${clipEnabled ? "is-active" : ""}`}
                title="Toggle clip plane (cut the volume at the slice plane)"
                onClick={() => setClipEnabled((value) => !value)}
                type="button"
              >
                <BoxSelect size={16} />
                <span>Clip</span>
              </button>
              <button
                className={`icon-toggle compact ${clipFlip ? "is-active" : ""}`}
                title="Reverse clip side (keep the other half)"
                onClick={() => setClipFlip((value) => !value)}
                disabled={!clipEnabled}
                type="button"
              >
                <FlipHorizontal2 size={16} />
                <span>Reverse</span>
              </button>
            </div>

            <button
              className={`icon-toggle compact ${intensityClipEnabled ? "is-active" : ""}`}
              title="Toggle intensity clip"
              onClick={() => setIntensityClipEnabled((value) => !value)}
              type="button"
            >
              <Scissors size={16} />
              <span>Intensity</span>
            </button>

            <label className="fps-control" title="Viewport FPS limit">
              <Gauge size={16} />
              <span>{fpsLimit} FPS</span>
              <input
                max={120}
                min={15}
                onChange={(event) => setFpsLimit(Number(event.currentTarget.value))}
                step={5}
                type="range"
                value={fpsLimit}
              />
            </label>
          </div>

          <GaussianViewport
            data={gaussians}
            session={session}
            renderMode={renderMode}
            axis={axis}
            sliceT={sliceT}
            surfaceAlpha={surfaceAlpha}
            bulkAlpha={bulkAlpha}
            sliceFadeWidthMm={sliceFadeWidthMm}
            clipSoftnessMm={clipSoftnessMm}
            clipEnabled={clipEnabled}
            clipFlip={clipFlip}
            fpsLimit={fpsLimit}
            splatRadiusScale={splatRadiusScale}
            intensityClipEnabled={intensityClipEnabled}
            intensityMin={intensityMin}
            intensityMax={intensityMax}
          />
        </section>

        <aside className="inspector-column">
          {/* Sticky top: slice preview + position slider always visible */}
          <div className="inspector-sticky">
            <SlicePanel
              axis={axis}
              imageUrl={sliceImageUrl}
              layer={sliceLayer}
              previewMode={slicePreviewMode}
              intensityClipEnabled={intensityClipEnabled}
              loading={sliceLoading}
              session={session}
              sliceT={deferredSliceT}
            />
            <div className="slice-slider-band">
              <div className="slider-label-row">
                <span><ScanLine size={14} /> Slice position</span>
                <span>{sliceT.toFixed(3)}</span>
              </div>
              <input
                className="wide-slider"
                max={1}
                min={0}
                onChange={(event) => {
                  const nextValue = Number(event.currentTarget.value);
                  startTransition(() => setSliceT(nextValue));
                }}
                step={0.001}
                type="range"
                value={sliceT}
              />
            </div>
          </div>

          {/* Scrollable parameter sections — ordered by inspection workflow:
              cut the volume, isolate by intensity, then tune appearance. */}
          <div className="inspector-scrollable">
            <section className="inspector-section">
              <div className="section-heading">
                <span><BoxSelect size={14} /> Clip Plane</span>
              </div>
              <div className="section-toolbar">
                <button
                  className={`icon-toggle compact ${clipEnabled ? "is-active" : ""}`}
                  onClick={() => setClipEnabled((value) => !value)}
                  title="Cut the volume at the slice plane"
                  type="button"
                >
                  <BoxSelect size={14} />
                  <span>{clipEnabled ? "Enabled" : "Disabled"}</span>
                </button>
                <button
                  className={`icon-toggle compact ${clipFlip ? "is-active" : ""}`}
                  onClick={() => setClipFlip((value) => !value)}
                  disabled={!clipEnabled}
                  title="Reverse clip side (keep the other half)"
                  type="button"
                >
                  <FlipHorizontal2 size={14} />
                  <span>Reverse</span>
                </button>
              </div>
              <label className="slider-field">
                <span>Clip softness</span>
                <span>{clipSoftnessMm.toFixed(2)}</span>
                <input
                  disabled={!clipEnabled}
                  max={Math.max(maxClipSoftness, 0.1)}
                  min={0.001}
                  onChange={(event) => setClipSoftnessMm(Number(event.currentTarget.value))}
                  step={0.01}
                  type="range"
                  value={clipSoftnessMm}
                />
              </label>
            </section>

            <section className="inspector-section">
              <div className="section-heading">
                <span><Scissors size={14} /> Intensity Clip</span>
                <button
                  className="icon-toggle compact mini"
                  onClick={resetIntensityClip}
                  title="Reset intensity clip"
                  type="button"
                >
                  <RotateCcw size={13} />
                </button>
              </div>
              <div className="section-toolbar">
                <button
                  className={`icon-toggle compact ${intensityClipEnabled ? "is-active" : ""}`}
                  onClick={() => setIntensityClipEnabled((value) => !value)}
                  title="Apply intensity clip"
                  type="button"
                >
                  <Scissors size={14} />
                  <span>{intensityClipEnabled ? "Enabled" : "Disabled"}</span>
                </button>
                <div className="segmented-group preview-mode-group" aria-label="Slice preview mode">
                  {PREVIEW_MODES.map((candidate) => (
                    <button
                      key={candidate}
                      className={slicePreviewMode === candidate ? "is-active" : ""}
                      onClick={() => setSlicePreviewMode(candidate)}
                      type="button"
                    >
                      {candidate}
                    </button>
                  ))}
                </div>
              </div>
              <label className="slider-field">
                <span>Min intensity</span>
                <span>{intensityMin.toFixed(3)}</span>
                <input
                  disabled={!intensityClipEnabled}
                  max={intensityDomainMax}
                  min={intensityDomainMin}
                  onChange={(event) => updateIntensityMin(Number(event.currentTarget.value))}
                  step={0.005}
                  type="range"
                  value={intensityMin}
                />
              </label>
              <label className="slider-field">
                <span>Max intensity</span>
                <span>{intensityMax.toFixed(3)}</span>
                <input
                  disabled={!intensityClipEnabled}
                  max={intensityDomainMax}
                  min={intensityDomainMin}
                  onChange={(event) => updateIntensityMax(Number(event.currentTarget.value))}
                  step={0.005}
                  type="range"
                  value={intensityMax}
                />
              </label>
              {surfaceIntensityStats !== undefined && bulkIntensityStats !== undefined ? (
                <div className="range-summary">
                  <span>S p50 {surfaceIntensityStats.p50.toFixed(3)}</span>
                  <span>B p50 {bulkIntensityStats.p50.toFixed(3)}</span>
                  <span>B p95 {bulkIntensityStats.p95.toFixed(3)}</span>
                </div>
              ) : null}
            </section>

            <section className="inspector-section">
              <div className="section-heading">
                <span><Columns3 size={14} /> Appearance</span>
              </div>
              <label className="slider-field">
                <span>Surface attenuation</span>
                <span>{surfaceAlpha.toFixed(2)}</span>
                <input max={1} min={0} onChange={(event) => setSurfaceAlpha(Number(event.currentTarget.value))} step={0.01} type="range" value={surfaceAlpha} />
              </label>
              <label className="slider-field">
                <span>Bulk attenuation</span>
                <span>{bulkAlpha.toFixed(2)}</span>
                <input max={1} min={0} onChange={(event) => setBulkAlpha(Number(event.currentTarget.value))} step={0.01} type="range" value={bulkAlpha} />
              </label>
              <label className="slider-field">
                <span>Splat radius</span>
                <span>{splatRadiusScale.toFixed(2)}×</span>
                <input
                  max={4}
                  min={0.1}
                  onChange={(event) => setSplatRadiusScale(Number(event.currentTarget.value))}
                  step={0.05}
                  type="range"
                  value={splatRadiusScale}
                />
              </label>
            </section>

            <section className="inspector-section">
              <div className="section-heading">
                <span><Waves size={14} /> Slice Rendering</span>
              </div>
              <label className="slider-field">
                <span>Slice fade width</span>
                <span>{sliceFadeWidthMm.toFixed(2)}</span>
                <input
                  max={Math.max(maxSliceFade, 0.5)}
                  min={0}
                  onChange={(event) => setSliceFadeWidthMm(Number(event.currentTarget.value))}
                  step={0.01}
                  type="range"
                  value={sliceFadeWidthMm}
                />
              </label>
            </section>

            <section className="inspector-section">
              <div className="section-heading">
                <span><Eye size={14} /> Session</span>
              </div>
              {session !== null ? (
                <dl className="session-grid">
                  <div>
                    <dt>PLY</dt>
                    <dd>{session.ply_path.split(/[\\/]/).slice(-1)[0]}</dd>
                  </div>
                  <div>
                    <dt>Gaussians</dt>
                    <dd>{session.gaussian_count.toLocaleString()} ({session.surface_count.toLocaleString()} surface · {session.bulk_count.toLocaleString()} bulk)</dd>
                  </div>
                  <div>
                    <dt>BBox</dt>
                    <dd>{session.bbox.size.map((v) => v.toFixed(2)).join(" × ")}</dd>
                  </div>
                  <div>
                    <dt>Stride</dt>
                    <dd>{gaussianMeta?.stride_bytes ?? 0} B · {gaussianMeta?.dtype ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>Device</dt>
                    <dd>{session.device}</dd>
                  </div>
                </dl>
              ) : null}
              {loadError !== null ? <p className="error-text">{loadError}</p> : null}
            </section>
          </div>
        </aside>
      </main>
    </div>
  );
}
