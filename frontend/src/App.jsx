import { NavLink, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Assets from './pages/Assets'
import AssetDetail from './pages/AssetDetail'
import LiveAnalysis from './pages/LiveAnalysis'
import SemanticMemory from './pages/SemanticMemory'
import MissionHistory from './pages/MissionHistory'

const NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/assets', label: 'Assets' },
  { to: '/analysis', label: 'Live Analysis' },
  { to: '/missions', label: 'Mission History' },
  { to: '/memory', label: 'Semantic Memory' },
]

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>AUDI</h1>
          <p>AI-Powered Persistent Semantic Memory<br />Drone Inspection Intelligence</p>
        </div>
        <nav>
          {NAV.map(n => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === '/'}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/assets" element={<Assets />} />
          <Route path="/assets/:id" element={<AssetDetail />} />
          <Route path="/analysis" element={<LiveAnalysis />} />
          <Route path="/missions" element={<MissionHistory />} />
          <Route path="/memory" element={<SemanticMemory />} />
        </Routes>
      </main>
    </div>
  )
}
