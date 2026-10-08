import { useState } from 'react'
import { analyzeImage } from '../api'
import ImageUpload from '../components/ImageUpload'
import AnalysisResult from '../components/AnalysisResult'

export default function LiveAnalysis() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleUpload = async (file, opts = {}) => {
    if (opts.clientError) {
      setError(opts.clientError)
      setResult(null)
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await analyzeImage(file, true)
      if (data.error) {
        setError(data.error)
      } else {
        setResult(data)
      }
    } catch (e) {
      const detail = e.response?.data?.detail
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail) || e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2 className="page-title">Live Analysis</h2>
      <p className="page-subtitle">Upload an inspection image for semantic memory analysis</p>

      <div className="banner">
        Pipeline: Image → Embedding → Vector Search → Asset Match → Change Detection → Severity → Revisit Priority
      </div>

      <ImageUpload onUpload={handleUpload} loading={loading} />
      {error && <div className="error-box">{error}</div>}
      <AnalysisResult result={result} />
    </div>
  )
}
