export type Prediction = {
  label: string
  confidence: number
}

export type ScanResult = {
  prediction: string
  confidence: number
  top_predictions: Prediction[]
  background_reliance: number
  model: string
  classes_covered: number
  gradcam_available: boolean
}

const backendUrl = process.env.NEXT_PUBLIC_AGRISIGHT_API_URL || 'http://localhost:8000'

function isScanResult(value: unknown): value is ScanResult {
  if (!value || typeof value !== 'object') return false
  const result = value as Partial<ScanResult>
  return typeof result.prediction === 'string' &&
    typeof result.confidence === 'number' && Number.isFinite(result.confidence) && result.confidence >= 0 && result.confidence <= 1 &&
    Array.isArray(result.top_predictions) && result.top_predictions.every((item) =>
      item && typeof item.label === 'string' && typeof item.confidence === 'number' && item.confidence >= 0 && item.confidence <= 1
    ) &&
    typeof result.background_reliance === 'number' &&
    typeof result.model === 'string' &&
    typeof result.classes_covered === 'number' &&
    typeof result.gradcam_available === 'boolean'
}

export async function scanCropImage(file: File, signal?: AbortSignal): Promise<ScanResult> {
  const body = new FormData()
  body.append('file', file)
  let response: Response
  try {
    response = await fetch(`${backendUrl.replace(/\/$/, '')}/predict`, { method: 'POST', body, signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new Error('BACKEND_UNAVAILABLE')
  }
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = payload && typeof payload === 'object' && 'detail' in payload && typeof payload.detail === 'string' ? payload.detail : null
    if (response.status === 400 || response.status === 415 || response.status === 413) throw new Error(detail || 'INVALID_IMAGE')
    if (response.status >= 500) throw new Error('BACKEND_UNAVAILABLE')
    throw new Error(detail || 'SCAN_FAILED')
  }
  if (!isScanResult(payload)) throw new Error('INVALID_RESPONSE')
  return payload
}
