import type { AxisName, SessionPayload, SliceLayer, SlicePreviewMode } from "../types";
import { computeSliceRequestSize } from "../slicePreview";

function axisCoordinate(axis: AxisName, session: SessionPayload, sliceT: number): number {
  if (axis === "x") return session.bbox.min[0] + sliceT * (session.bbox.max[0] - session.bbox.min[0]);
  if (axis === "y") return session.bbox.min[1] + sliceT * (session.bbox.max[1] - session.bbox.min[1]);
  return session.bbox.min[2] + sliceT * (session.bbox.max[2] - session.bbox.min[2]);
}

export function SlicePanel(props: {
  session: SessionPayload | null;
  axis: AxisName;
  sliceT: number;
  layer: SliceLayer;
  previewMode: SlicePreviewMode;
  intensityClipEnabled: boolean;
  loading: boolean;
  imageUrl: string | null;
}) {
  const coordinate = props.session ? axisCoordinate(props.axis, props.session, props.sliceT) : 0.0;
  const baseSliceSize = props.session?.defaults.sliceSize ?? 512;
  const deviceScale = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
  const sliceSize = computeSliceRequestSize(baseSliceSize, deviceScale);
  const imageRendering = sliceSize <= 128 ? "pixelated" : "auto";

  return (
    <section className="slice-panel">
      <div className="section-heading">
        <span>Slice Preview</span>
        <span className="section-meta">
          {props.axis.toUpperCase()} {coordinate.toFixed(3)} / {props.layer} / {props.intensityClipEnabled ? props.previewMode : "raw"} / {sliceSize}px
        </span>
      </div>
      <div className="slice-stage">
        {props.imageUrl !== null ? (
          <img
            src={props.imageUrl}
            alt="CTGS generated slice"
            className="slice-image"
            style={{ imageRendering }}
          />
        ) : null}
        {props.loading ? <div className="slice-overlay">Updating...</div> : null}
        {props.imageUrl === null && !props.loading ? (
          <div className="slice-overlay">Slice unavailable</div>
        ) : null}
      </div>
    </section>
  );
}
