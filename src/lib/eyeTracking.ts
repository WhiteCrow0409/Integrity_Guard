import { FilesetResolver, FaceLandmarker, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';

let landmarker: FaceLandmarker | null = null;

export type GazeDirection = 'center' | 'left' | 'right' | 'up' | 'down';

export interface EyeMetrics {
  blinkLeft: boolean;
  blinkRight: boolean;
  blinkScoreLeft: number;
  blinkScoreRight: number;
  gaze: GazeDirection;
}

export async function initEyeTracker() {
  if (landmarker) return landmarker;
  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );
    // Use the .task packaged model to ensure blendshapes are available.
    landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false
    });
    return landmarker;
  } catch (err) {
    console.error('Eye tracker initialization failed:', err);
    landmarker = null;
    return null;
  }
}

function getBlendshapeScore(res: FaceLandmarkerResult, name: string): number {
  const shapes = res.faceBlendshapes?.[0]?.categories;
  if (!shapes) return 0;
  const item = shapes.find((c) => c.categoryName === name);
  return item?.score ?? 0;
}

function inferGaze(res: FaceLandmarkerResult): GazeDirection {
  // Aggregate left/right eye look blendshapes
  const lookLeft =
    getBlendshapeScore(res, 'eyeLookOutLeft') + getBlendshapeScore(res, 'eyeLookInRight');
  const lookRight =
    getBlendshapeScore(res, 'eyeLookInLeft') + getBlendshapeScore(res, 'eyeLookOutRight');
  const lookUp =
    getBlendshapeScore(res, 'eyeLookUpLeft') + getBlendshapeScore(res, 'eyeLookUpRight');
  const lookDown =
    getBlendshapeScore(res, 'eyeLookDownLeft') + getBlendshapeScore(res, 'eyeLookDownRight');

  const maxVal = Math.max(lookLeft, lookRight, lookUp, lookDown);
  const threshold = 0.6; // lowered to make gaze detection more sensitive
  if (maxVal < threshold) return 'center';
  if (maxVal === lookLeft) return 'left';
  if (maxVal === lookRight) return 'right';
  if (maxVal === lookUp) return 'up';
  return 'down';
}

export async function detectEyeMetrics(
  video: HTMLVideoElement
): Promise<EyeMetrics | null> {
  if (!landmarker) await initEyeTracker();
  if (!landmarker) return null;
  let res: FaceLandmarkerResult | null = null;
  try {
    res = await landmarker.detectForVideo(video, performance.now());
  } catch (err) {
    console.error('Eye tracker detectForVideo failed:', err);
    return null;
  }
  if (!res) return null;
  if (!res.faceBlendshapes || res.faceBlendshapes.length === 0) {
    // Minimal fallback: if we have no blendshapes, return center gaze with no blinks
    return {
      blinkLeft: false,
      blinkRight: false,
      blinkScoreLeft: 0,
      blinkScoreRight: 0,
      gaze: 'center'
    };
  }

  const blinkLeftScore = getBlendshapeScore(res, 'eyeBlinkLeft');
  const blinkRightScore = getBlendshapeScore(res, 'eyeBlinkRight');
  const blinkThreshold = 0.5;

  return {
    blinkLeft: blinkLeftScore > blinkThreshold,
    blinkRight: blinkRightScore > blinkThreshold,
    blinkScoreLeft: blinkLeftScore,
    blinkScoreRight: blinkRightScore,
    gaze: inferGaze(res)
  };
}

export function cleanupEyeTracker() {
  landmarker = null;
}
