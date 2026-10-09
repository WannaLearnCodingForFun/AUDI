// Frontend tuning (the backend's equivalents live in config.yaml).
export const BROWSER_CAMERA = {
  frameIntervalMs: 200,      // browser-camera mode: one JPEG frame sent per interval (5 fps)
  jpegQuality: 0.8,
  width: 1280,
  height: 720,
  maxBufferedBytes: 2_000_000, // skip a frame while the socket is still sending the previous ones
}

export const ALERTS = {
  maxShown: 25,              // alert list length on the live screen
  bannerMs: 6000,            // edge banner visible for this long after the latest alert
  priorityTone: { freq: 880, ms: 180, repeats: 1 },
  criticalTone: { freq: 1320, ms: 140, repeats: 3 },
}

export const STREAM_RETRY_MS = 2000 // MJPEG <img> reload delay after an error
