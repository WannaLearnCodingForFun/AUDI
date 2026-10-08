import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getAssets } from '../api'
import AssetTable from '../components/AssetTable'

export default function Assets() {
  const [assets, setAssets] = useState([])
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    getAssets()
      .then(d => setAssets(d.assets || []))
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div className="error-box">{error}</div>

  return (
    <div>
      <h2 className="page-title">Assets</h2>
      <p className="page-subtitle">All registered solar panel assets in semantic memory</p>
      <AssetTable assets={assets} onSelect={id => navigate(`/assets/${id}`)} />
    </div>
  )
}
