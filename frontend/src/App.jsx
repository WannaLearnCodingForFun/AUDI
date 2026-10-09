import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { hasRole, useAuth } from './auth'
import Layout from './components/Layout'
import { Loading } from './components/ui'
import AdminConfig from './pages/AdminConfig'
import AdminMetrics from './pages/AdminMetrics'
import AdminUsers from './pages/AdminUsers'
import AssetDetail from './pages/AssetDetail'
import Assets from './pages/Assets'
import Briefing from './pages/Briefing'
import Dashboard from './pages/Dashboard'
import DefectDetail from './pages/DefectDetail'
import Defects from './pages/Defects'
import Inbox from './pages/Inbox'
import LiveFlight from './pages/LiveFlight'
import Login from './pages/Login'
import MissionHistory from './pages/MissionHistory'
import Report from './pages/Report'
import SemanticMemory from './pages/SemanticMemory'
import Sites from './pages/Sites'

function Guard({ roles, children }) {
  const { user } = useAuth()
  if (!hasRole(user, ...roles)) return <div className="pad">This page needs the {roles.join(' or ') || 'admin'} role.</div>
  return children
}

const home = (user) => (user.role === 'admin' ? '/admin/metrics' : '/')

export default function App() {
  const { user, ready } = useAuth()
  const loc = useLocation()
  if (!ready) return <Loading what="Starting" />
  if (!user) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<Navigate to="/login" state={{ from: loc.pathname }} replace />} />
      </Routes>
    )
  }
  const field = ['operator', 'maintenance']
  return (
    <Routes>
      <Route path="/login" element={<Navigate to={home(user)} replace />} />
      <Route element={<Layout />}>
        <Route index element={<Guard roles={field}><Dashboard /></Guard>} />
        <Route path="sites" element={<Guard roles={['operator']}><Sites /></Guard>} />
        <Route path="briefing/:siteId" element={<Guard roles={['operator']}><Briefing /></Guard>} />
        <Route path="flight/:sessionId" element={<Guard roles={['operator']}><LiveFlight /></Guard>} />
        <Route path="report/:sessionId" element={<Guard roles={field}><Report /></Guard>} />
        <Route path="missions" element={<Guard roles={field}><MissionHistory /></Guard>} />
        <Route path="inbox" element={<Guard roles={['maintenance']}><Inbox /></Guard>} />
        <Route path="defects" element={<Guard roles={field}><Defects /></Guard>} />
        <Route path="defects/:defectId" element={<Guard roles={field}><DefectDetail /></Guard>} />
        <Route path="assets" element={<Guard roles={field}><Assets /></Guard>} />
        <Route path="assets/:assetId" element={<Guard roles={field}><AssetDetail /></Guard>} />
        <Route path="memory" element={<Guard roles={field}><SemanticMemory /></Guard>} />
        <Route path="admin/config" element={<Guard roles={[]}><AdminConfig /></Guard>} />
        <Route path="admin/users" element={<Guard roles={[]}><AdminUsers /></Guard>} />
        <Route path="admin/metrics" element={<Guard roles={[]}><AdminMetrics /></Guard>} />
        <Route path="*" element={<Navigate to={home(user)} replace />} />
      </Route>
    </Routes>
  )
}
