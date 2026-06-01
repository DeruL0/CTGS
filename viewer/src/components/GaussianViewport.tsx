import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";

import type { AxisName, GaussianBuffer, RenderMode, SessionPayload } from "../types";

const MODE_INDEX: Record<RenderMode, number> = {
  composite: 0,
  "surface-lit": 1,
  "surface-normal": 2,
  region: 3,
  "bulk-only": 4,
  intensity: 5
};

const AXIS_NORMAL: Record<AxisName, [number, number, number]> = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1]
};

// Number of stored-sigma the quad covers in each direction.
// Fixed constant ensures the Gaussian always decays to ~0 at quad edges
// regardless of uSplatRadiusScale, eliminating hard rectangular edges.
const SIGMA_EXTENT = "3.5";
const DEPTH_SORT_INTERVAL_MS = 80;
const DEPTH_SORT_POSITION_EPS_SQ = 1e-6;
const DEPTH_SORT_DIRECTION_EPS = 1e-5;

interface DepthSortState {
  order: Uint32Array;
  depths: Float32Array;
  center: THREE.InstancedBufferAttribute;
  scale: THREE.InstancedBufferAttribute;
  rotation: THREE.InstancedBufferAttribute;
  normal: THREE.InstancedBufferAttribute;
  opacity: THREE.InstancedBufferAttribute;
  region: THREE.InstancedBufferAttribute;
  supportRadius: THREE.InstancedBufferAttribute;
  intensity: THREE.InstancedBufferAttribute;
}

function makeOrder(count: number): Uint32Array {
  const order = new Uint32Array(count);
  for (let index = 0; index < count; index += 1) {
    order[index] = index;
  }
  return order;
}

function copySortedComponents(source: Float32Array, target: Float32Array, order: Uint32Array, itemSize: number) {
  for (let sortedIndex = 0; sortedIndex < order.length; sortedIndex += 1) {
    const sourceOffset = order[sortedIndex] * itemSize;
    const targetOffset = sortedIndex * itemSize;
    for (let component = 0; component < itemSize; component += 1) {
      target[targetOffset + component] = source[sourceOffset + component];
    }
  }
}

function markAttributesDirty(state: DepthSortState) {
  state.center.needsUpdate = true;
  state.scale.needsUpdate = true;
  state.rotation.needsUpdate = true;
  state.normal.needsUpdate = true;
  state.opacity.needsUpdate = true;
  state.region.needsUpdate = true;
  state.supportRadius.needsUpdate = true;
  state.intensity.needsUpdate = true;
}

function sortGaussianAttributes(
  data: GaussianBuffer,
  state: DepthSortState,
  cameraPosition: THREE.Vector3,
  cameraDirection: THREE.Vector3
) {
  const positions = data.positions;
  for (let index = 0; index < data.count; index += 1) {
    const offset = index * 3;
    const dx = positions[offset] - cameraPosition.x;
    const dy = positions[offset + 1] - cameraPosition.y;
    const dz = positions[offset + 2] - cameraPosition.z;
    state.depths[index] = dx * cameraDirection.x + dy * cameraDirection.y + dz * cameraDirection.z;
  }

  state.order.sort((left, right) => state.depths[right] - state.depths[left]);
  copySortedComponents(data.positions, state.center.array as Float32Array, state.order, 3);
  copySortedComponents(data.scales, state.scale.array as Float32Array, state.order, 3);
  copySortedComponents(data.rotations, state.rotation.array as Float32Array, state.order, 4);
  copySortedComponents(data.normals, state.normal.array as Float32Array, state.order, 3);
  copySortedComponents(data.opacity, state.opacity.array as Float32Array, state.order, 1);
  copySortedComponents(data.regionType, state.region.array as Float32Array, state.order, 1);
  copySortedComponents(data.supportRadius, state.supportRadius.array as Float32Array, state.order, 1);
  copySortedComponents(data.intensity, state.intensity.array as Float32Array, state.order, 1);
  markAttributesDirty(state);
}

function planeRotation(axis: AxisName): [number, number, number] {
  if (axis === "x") return [0, Math.PI / 2, 0];
  if (axis === "y") return [Math.PI / 2, 0, 0];
  return [0, 0, 0];
}

function planeSize(axis: AxisName, bboxSize: [number, number, number]): [number, number] {
  if (axis === "x") return [bboxSize[1], bboxSize[2]];
  if (axis === "y") return [bboxSize[0], bboxSize[2]];
  return [bboxSize[0], bboxSize[1]];
}

function planePosition(axis: AxisName, bbox: SessionPayload["bbox"], sliceT: number): [number, number, number] {
  const x = bbox.min[0] + sliceT * (bbox.max[0] - bbox.min[0]);
  const y = bbox.min[1] + sliceT * (bbox.max[1] - bbox.min[1]);
  const z = bbox.min[2] + sliceT * (bbox.max[2] - bbox.min[2]);
  if (axis === "x") return [x, bbox.center[1], bbox.center[2]];
  if (axis === "y") return [bbox.center[0], y, bbox.center[2]];
  return [bbox.center[0], bbox.center[1], z];
}

function sliceCoordinate(axis: AxisName, bbox: SessionPayload["bbox"], sliceT: number): number {
  if (axis === "x") return bbox.min[0] + sliceT * (bbox.max[0] - bbox.min[0]);
  if (axis === "y") return bbox.min[1] + sliceT * (bbox.max[1] - bbox.min[1]);
  return bbox.min[2] + sliceT * (bbox.max[2] - bbox.min[2]);
}

// Calls invalidate() at a throttled rate so frameloop="demand" re-renders
// continuously at the desired FPS. OrbitControls damping also calls invalidate
// internally, so damped inertia renders correctly without fighting this timer.
function FrameThrottle({ fpsLimit }: { fpsLimit: number }) {
  const invalidate = useThree((state) => state.invalidate);
  useEffect(() => {
    const id = setInterval(invalidate, 1000 / Math.max(1, fpsLimit));
    return () => clearInterval(id);
  }, [invalidate, fpsLimit]);
  return null;
}

function GaussianSplats(props: {
  data: GaussianBuffer;
  session: SessionPayload;
  renderMode: RenderMode;
  axis: AxisName;
  sliceT: number;
  surfaceAlpha: number;
  bulkAlpha: number;
  sliceFadeWidthMm: number;
  clipSoftnessMm: number;
  clipEnabled: boolean;
  clipFlip: boolean;
  splatRadiusScale: number;
  intensityClipEnabled: boolean;
  intensityMin: number;
  intensityMax: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  const sortStateRef = useRef<DepthSortState | null>(null);
  const cameraPositionRef = useRef(new THREE.Vector3());
  const cameraDirectionRef = useRef(new THREE.Vector3());
  const lastSortRef = useRef({
    initialized: false,
    timeMs: 0,
    position: new THREE.Vector3(),
    direction: new THREE.Vector3()
  });
  const bbox = props.session.bbox;

  useFrame(({ camera, clock }) => {
    const sortState = sortStateRef.current;
    if (sortState === null) return;
    if (props.renderMode === "composite") return;

    const nowMs = clock.elapsedTime * 1000.0;
    const cameraPosition = cameraPositionRef.current;
    const cameraDirection = cameraDirectionRef.current;
    camera.getWorldPosition(cameraPosition);
    camera.getWorldDirection(cameraDirection);

    const lastSort = lastSortRef.current;
    const moved =
      !lastSort.initialized ||
      cameraPosition.distanceToSquared(lastSort.position) > DEPTH_SORT_POSITION_EPS_SQ ||
      1.0 - cameraDirection.dot(lastSort.direction) > DEPTH_SORT_DIRECTION_EPS;
    if (!moved) return;
    if (lastSort.initialized && nowMs - lastSort.timeMs < DEPTH_SORT_INTERVAL_MS) return;

    sortGaussianAttributes(props.data, sortState, cameraPosition, cameraDirection);
    lastSort.initialized = true;
    lastSort.timeMs = nowMs;
    lastSort.position.copy(cameraPosition);
    lastSort.direction.copy(cameraDirection);
  });

  useEffect(() => {
    if (meshRef.current === null) return undefined;

    const baseGeometry = new THREE.PlaneGeometry(1, 1, 1, 1);
    const geometry = new THREE.InstancedBufferGeometry();
    const centerAttribute = new THREE.InstancedBufferAttribute(new Float32Array(props.data.positions), 3);
    const scaleAttribute = new THREE.InstancedBufferAttribute(new Float32Array(props.data.scales), 3);
    const rotationAttribute = new THREE.InstancedBufferAttribute(new Float32Array(props.data.rotations), 4);
    const normalAttribute = new THREE.InstancedBufferAttribute(new Float32Array(props.data.normals), 3);
    const opacityAttribute = new THREE.InstancedBufferAttribute(new Float32Array(props.data.opacity), 1);
    const regionAttribute = new THREE.InstancedBufferAttribute(new Float32Array(props.data.regionType), 1);
    const supportRadiusAttribute = new THREE.InstancedBufferAttribute(new Float32Array(props.data.supportRadius), 1);
    const intensityAttribute = new THREE.InstancedBufferAttribute(new Float32Array(props.data.intensity), 1);
    geometry.index = baseGeometry.index;
    geometry.setAttribute("position", baseGeometry.getAttribute("position"));
    geometry.setAttribute("uv", baseGeometry.getAttribute("uv"));
    geometry.instanceCount = props.data.count;
    geometry.setAttribute("instanceCenter", centerAttribute);
    geometry.setAttribute("instanceScale", scaleAttribute);
    geometry.setAttribute("instanceQuat", rotationAttribute);
    geometry.setAttribute("instanceNormal", normalAttribute);
    geometry.setAttribute("instanceOpacity", opacityAttribute);
    geometry.setAttribute("instanceRegion", regionAttribute);
    geometry.setAttribute("instanceSupportRadius", supportRadiusAttribute);
    geometry.setAttribute("instanceIntensity", intensityAttribute);
    sortStateRef.current = {
      order: makeOrder(props.data.count),
      depths: new Float32Array(props.data.count),
      center: centerAttribute,
      scale: scaleAttribute,
      rotation: rotationAttribute,
      normal: normalAttribute,
      opacity: opacityAttribute,
      region: regionAttribute,
      supportRadius: supportRadiusAttribute,
      intensity: intensityAttribute
    };
    lastSortRef.current.initialized = false;

    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      uniforms: {
        uMode: { value: MODE_INDEX[props.renderMode] },
        uSurfaceAlpha: { value: props.surfaceAlpha },
        uBulkAlpha: { value: props.bulkAlpha },
        uPlaneNormal: { value: new THREE.Vector3(...AXIS_NORMAL[props.axis]) },
        uSliceCoord: { value: sliceCoordinate(props.axis, bbox, props.sliceT) },
        uSliceFadeWidth: { value: props.sliceFadeWidthMm },
        uClipSoftness: { value: props.clipSoftnessMm },
        uClipEnabled: { value: props.clipEnabled ? 1.0 : 0.0 },
        uClipFlip: { value: props.clipFlip ? 1.0 : 0.0 },
        uSplatRadiusScale: { value: props.splatRadiusScale },
        uIntensityClipEnabled: { value: props.intensityClipEnabled ? 1.0 : 0.0 },
        uIntensityMin: { value: props.intensityMin },
        uIntensityMax: { value: props.intensityMax }
      },
      vertexShader: `
        attribute vec3 instanceCenter;
        attribute vec3 instanceScale;
        attribute vec4 instanceQuat;
        attribute vec3 instanceNormal;
        attribute float instanceOpacity;
        attribute float instanceRegion;
        attribute float instanceSupportRadius;
        attribute float instanceIntensity;

        uniform vec3 uPlaneNormal;
        uniform float uSliceCoord;
        uniform float uSplatRadiusScale;

        varying vec2 vUvCentered;
        varying vec3 vNormal;
        varying float vOpacity;
        varying float vRegion;
        varying float vCenterPlaneDistance;
        varying vec3 vWorldPos;
        varying float vIntensity;

        vec3 quatRotate(vec4 q, vec3 v) {
          return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
        }

        void main() {
          const float SE = ${SIGMA_EXTENT};
          float radiusScale = max(uSplatRadiusScale, 0.01);
          // Quad always covers SE sigma of the scaled Gaussian — guarantees
          // The quad spans +/-SE sigma and decays to near zero at the edge.
          // Viewport display uses normal GS-style camera-facing splats. Normals
          // only affect surface lighting/debug color, not quad orientation.
          vec4 q = normalize(instanceQuat);
          vec3 cameraRight = normalize(vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]));
          vec3 cameraUp = normalize(vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]));
          vec3 localRight = quatRotate(vec4(-q.xyz, q.w), cameraRight);
          vec3 localUp = quatRotate(vec4(-q.xyz, q.w), cameraUp);
          float rightSigma = length(localRight * instanceScale);
          float upSigma = length(localUp * instanceScale);
          if (instanceRegion >= 0.5 && instanceSupportRadius > 0.0) {
            float compactSigmaCap = max(instanceSupportRadius / SE, 0.000001);
            rightSigma = min(rightSigma, compactSigmaCap);
            upSigma = min(upSigma, compactSigmaCap);
          }
          vec3 worldPosition =
            instanceCenter
            + cameraRight * position.x * rightSigma * radiusScale * 2.0 * SE
            + cameraUp * position.y * upSigma * radiusScale * 2.0 * SE;

          vUvCentered = uv * 2.0 - 1.0;
          vNormal = normalize(instanceNormal);
          vOpacity = instanceOpacity;
          vRegion = instanceRegion;
          vIntensity = instanceIntensity;
          // Gaussian-center distance used for slice fade (smooth, per-gaussian)
          vCenterPlaneDistance = dot(instanceCenter, uPlaneNormal) - uSliceCoord;
          // Fragment world position for per-pixel clip plane evaluation
          vWorldPos = worldPosition;

          gl_Position = projectionMatrix * modelViewMatrix * vec4(worldPosition, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uMode;
        uniform float uSurfaceAlpha;
        uniform float uBulkAlpha;
        uniform float uSliceFadeWidth;
        uniform float uClipSoftness;
        uniform float uClipEnabled;
        uniform float uClipFlip;
        uniform vec3 uPlaneNormal;
        uniform float uSliceCoord;
        uniform float uIntensityClipEnabled;
        uniform float uIntensityMin;
        uniform float uIntensityMax;

        varying vec2 vUvCentered;
        varying vec3 vNormal;
        varying float vOpacity;
        varying float vRegion;
        varying float vCenterPlaneDistance;
        varying vec3 vWorldPos;
        varying float vIntensity;

        void main() {
          const float SE = ${SIGMA_EXTENT};
          // vUvCentered ∈ [-1,1] → sigmaCoord ∈ [-SE, SE].
          // At any radiusScale, corner radial = exp(-0.5·SE²·2) ≈ 0 → always discarded.
          vec2 sigmaCoord = vUvCentered * SE;
          float radial = exp(-0.5 * dot(sigmaCoord, sigmaCoord));
          if (radial < 0.002) discard;

          bool isSurface = vRegion < 0.5;

          // mode 1,2: surface only; mode 4: bulk only
          if (uMode > 0.5 && uMode < 2.5 && !isSurface) discard;
          if (uMode > 3.5 && uMode < 4.5 && isSurface) discard;

          if (uIntensityClipEnabled > 0.5) {
            float lo = min(uIntensityMin, uIntensityMax);
            float hi = max(uIntensityMin, uIntensityMax);
            if (vIntensity < lo || vIntensity > hi) discard;
          }

          float layerAttenuation = isSurface ? uSurfaceAlpha : uBulkAlpha;

          // Slice fade: per-gaussian-center distance, smooth exponential
          float sliceFade = 1.0;
          if (uSliceFadeWidth > 0.0) {
            float focus = exp(-abs(vCenterPlaneDistance) / max(uSliceFadeWidth, 0.0001));
            sliceFade = mix(0.10, 1.0, clamp(focus, 0.0, 1.0));
          }

          // Clip: per-fragment world position — sharp, pixel-accurate boundary
          float clipFactor = 1.0;
          if (uClipEnabled > 0.5) {
            // uClipFlip flips which half-space is kept (reverse clip).
            float clipSide = uClipFlip > 0.5 ? -1.0 : 1.0;
            float fragDist = clipSide * (dot(vWorldPos, uPlaneNormal) - uSliceCoord);
            float softness = max(uClipSoftness, 0.0001);
            if (fragDist < -softness) discard;
            clipFactor = smoothstep(-softness, 0.0, fragDist);
          }

          float tau = radial * vOpacity * layerAttenuation * sliceFade * clipFactor;
          if (tau < 0.002) discard;

          if (uMode < 0.5) {
            float attenuation = 1.0 - exp(-tau);
            if (attenuation < 0.002) discard;
            gl_FragColor = vec4(vec3(1.0), attenuation);
            return;
          }

          vec3 lightDir = normalize(vec3(0.32, 0.78, 0.54));
          float lambert = 0.32 + 0.68 * max(dot(normalize(vNormal), lightDir), 0.0);
          vec3 color;

          if (uMode < 1.5) {
            color = vec3(0.84, 0.90, 0.92) * lambert;
          } else if (uMode < 2.5) {
            color = normalize(vNormal) * 0.5 + 0.5;
          } else if (uMode < 4.5) {
            // region (3) or bulk-only (4)
            color = isSurface ? vec3(0.22, 0.73, 0.67) : vec3(0.89, 0.39, 0.22);
          } else {
            float x = clamp(vIntensity, 0.0, 1.0);
            vec3 low = vec3(0.07, 0.10, 0.18);
            vec3 mid = vec3(0.16, 0.68, 0.72);
            vec3 high = vec3(1.00, 0.83, 0.45);
            color = mix(low, mid, smoothstep(0.00, 0.55, x));
            color = mix(color, high, smoothstep(0.45, 1.00, x));
            color *= isSurface ? (0.72 + 0.28 * lambert) : 1.0;
          }

          float alpha = clamp(tau, 0.0, 1.0);
          gl_FragColor = vec4(color, alpha);
        }
      `
    });

    meshRef.current.geometry = geometry;
    meshRef.current.material = material;
    materialRef.current = material;

    return () => {
      sortStateRef.current = null;
      lastSortRef.current.initialized = false;
      geometry.dispose();
      material.dispose();
      baseGeometry.dispose();
    };
  }, [bbox, props.data]);

  useEffect(() => {
    if (materialRef.current === null) return;
    materialRef.current.uniforms.uMode.value = MODE_INDEX[props.renderMode];
    materialRef.current.uniforms.uSurfaceAlpha.value = props.surfaceAlpha;
    materialRef.current.uniforms.uBulkAlpha.value = props.bulkAlpha;
    materialRef.current.uniforms.uPlaneNormal.value.set(...AXIS_NORMAL[props.axis]);
    materialRef.current.uniforms.uSliceCoord.value = sliceCoordinate(props.axis, bbox, props.sliceT);
    materialRef.current.uniforms.uSliceFadeWidth.value = props.sliceFadeWidthMm;
    materialRef.current.uniforms.uClipSoftness.value = props.clipSoftnessMm;
    materialRef.current.uniforms.uClipEnabled.value = props.clipEnabled ? 1.0 : 0.0;
    materialRef.current.uniforms.uClipFlip.value = props.clipFlip ? 1.0 : 0.0;
    materialRef.current.uniforms.uSplatRadiusScale.value = props.splatRadiusScale;
    materialRef.current.uniforms.uIntensityClipEnabled.value = props.intensityClipEnabled ? 1.0 : 0.0;
    materialRef.current.uniforms.uIntensityMin.value = props.intensityMin;
    materialRef.current.uniforms.uIntensityMax.value = props.intensityMax;
  }, [bbox, props.axis, props.bulkAlpha, props.clipEnabled, props.clipFlip, props.clipSoftnessMm, props.intensityClipEnabled, props.intensityMax, props.intensityMin, props.renderMode, props.sliceFadeWidthMm, props.sliceT, props.splatRadiusScale, props.surfaceAlpha]);

  return <mesh ref={meshRef} frustumCulled={false} renderOrder={0} />;
}

export function GaussianViewport(props: {
  data: GaussianBuffer | null;
  session: SessionPayload | null;
  renderMode: RenderMode;
  axis: AxisName;
  sliceT: number;
  surfaceAlpha: number;
  bulkAlpha: number;
  sliceFadeWidthMm: number;
  clipSoftnessMm: number;
  clipEnabled: boolean;
  clipFlip: boolean;
  fpsLimit: number;
  splatRadiusScale: number;
  intensityClipEnabled: boolean;
  intensityMin: number;
  intensityMax: number;
}) {
  if (props.data === null || props.session === null) {
    return (
      <div className="viewport-shell viewport-loading">
        {props.session === null ? "Open a PLY to start." : "Loading viewer scene..."}
      </div>
    );
  }

  const bbox = props.session.bbox;
  const maxExtent = Math.max(...bbox.size);
  const cameraDistance = maxExtent * 2.8;
  const planeDims = planeSize(props.axis, bbox.size);
  const planePos = planePosition(props.axis, bbox, props.sliceT);
  const planeRot = planeRotation(props.axis);

  return (
    <div className="viewport-shell">
      <Canvas
        camera={{
          position: [
            bbox.center[0] + cameraDistance * 0.6,
            bbox.center[1] - cameraDistance,
            bbox.center[2] + cameraDistance * 0.8
          ],
          fov: 36,
          near: 0.01,
          far: cameraDistance * 12.0
        }}
        frameloop="demand"
      >
        <FrameThrottle fpsLimit={props.fpsLimit} />
        <color attach="background" args={["#000000"]} />
        <ambientLight intensity={0.55} />
        <directionalLight position={[maxExtent, maxExtent * 1.5, maxExtent]} intensity={1.15} />
        <mesh position={bbox.center} renderOrder={1}>
          <boxGeometry args={bbox.size} />
          <meshBasicMaterial color="#243140" wireframe transparent opacity={0.35} />
        </mesh>
        {/* Wireframe slice plane is an inspection overlay, not detector signal. */}
        <mesh position={planePos} renderOrder={1} rotation={planeRot}>
          <planeGeometry args={planeDims} />
          <meshBasicMaterial color="#78d6c4" wireframe transparent opacity={0.7} />
        </mesh>
        <GaussianSplats {...props} data={props.data} session={props.session} />
        <OrbitControls target={bbox.center} enableDamping />
      </Canvas>
    </div>
  );
}
