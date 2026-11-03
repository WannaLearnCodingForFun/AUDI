import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { getTwin } from '../api'
import { Card, ErrorBox, fmt, Loading } from '../components/ui'
import { useAsync } from '../hooks'

const BAND = { good: 0x3fb950, fair: 0xd29922, poor: 0xdb6d28, critical: 0xf85149, unknown: 0x8b949e }

/** 4.4 digital twin: assets at their ENU positions coloured by health; click a defect → its history. */
export default function SiteTwin() {
  const { siteId } = useParams()
  const twin = useAsync(() => getTwin(siteId), [siteId])
  const mount = useRef(null)
  const nav = useNavigate()
  const [hover, setHover] = useState(null)

  useEffect(() => {
    const scene = twin.data
    const el = mount.current
    if (!scene || !el) return undefined
    const w = el.clientWidth, h = 520
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(w, h)
    el.appendChild(renderer.domElement)
    const s = new THREE.Scene()
    s.background = new THREE.Color(0x0d1117)
    const cam = new THREE.PerspectiveCamera(50, w / h, 0.1, 5000)
    cam.up.set(0, 0, 1)
    cam.position.set(-40, -60, 50)
    const ctl = new OrbitControls(cam, renderer.domElement)
    s.add(new THREE.AmbientLight(0xffffff, 0.6))
    const sun = new THREE.DirectionalLight(0xffffff, 0.9)
    sun.position.set(50, -30, 80)
    s.add(sun)
    const grid = new THREE.GridHelper(400, 40, 0x30363d, 0x21262d)
    grid.rotation.x = Math.PI / 2
    s.add(grid)
    const pickables = []
    let cx = 0, cy = 0
    for (const a of scene.assets) {
      const [dx, dy, dz] = a.dims
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(dx, dy, dz), new THREE.MeshStandardMaterial({ color: BAND[a.health_band] || BAND.unknown }))
      mesh.position.set(a.position[0], a.position[1], dz / 2)
      mesh.userData = { kind: 'asset', id: a.asset_id, label: `${a.asset_id} · health ${fmt(a.health_index, 0)} · ${a.condition}` }
      s.add(mesh); pickables.push(mesh)
      cx += a.position[0]; cy += a.position[1]
      for (const d of a.defects) {
        const m = new THREE.Mesh(new THREE.SphereGeometry(Math.max(0.08, Math.min(dx, dy, dz) * 0.15 + 0.1), 16, 12), new THREE.MeshStandardMaterial({ color: 0xff3b6b, emissive: 0x550011 }))
        m.position.set(a.position[0] + d.local[0], a.position[1] + d.local[1], d.local[2] + 0.02)
        m.userData = { kind: 'defect', id: d.instance_id, label: `${d.defect_type} on ${a.asset_id}${d.predicted_critical_ts ? ` · critical ${new Date(d.predicted_critical_ts).toLocaleDateString()}` : ''}` }
        s.add(m); pickables.push(m)
      }
    }
    if (scene.assets.length) ctl.target.set(cx / scene.assets.length, cy / scene.assets.length, 0)
    const ray = new THREE.Raycaster()
    const ptr = new THREE.Vector2()
    const pick = (e) => {
      const r = renderer.domElement.getBoundingClientRect()
      ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
      ray.setFromCamera(ptr, cam)
      return ray.intersectObjects(pickables)[0]?.object?.userData
    }
    const onMove = (e) => setHover(pick(e)?.label || null)
    const onClick = (e) => {
      const u = pick(e)
      if (u?.kind === 'defect') nav(`/defect-instances/${u.id}`)
      else if (u?.kind === 'asset') nav(`/assets/${u.id}`)
    }
    renderer.domElement.addEventListener('pointermove', onMove)
    renderer.domElement.addEventListener('click', onClick)
    let raf
    const loop = () => { ctl.update(); renderer.render(s, cam); raf = requestAnimationFrame(loop) }
    loop()
    return () => { cancelAnimationFrame(raf); renderer.dispose(); el.removeChild(renderer.domElement) }
  }, [twin.data, nav])

  if (twin.loading) return <Loading />
  return (
    <div className="page">
      <header className="page-head"><h2>Digital twin · {siteId}</h2><span className="muted small">{hover || 'drag to orbit · click a red marker to open the defect'}</span></header>
      <ErrorBox error={twin.error} />
      <Card>
        <div ref={mount} className="twin" />
        {twin.data?.unlocated_assets?.length > 0 && <div className="small muted">{twin.data.unlocated_assets.length} assets without a location are not shown.</div>}
        {twin.data?.mesh && <div className="small muted">Photogrammetry mesh: <span className="mono">{twin.data.mesh.path}</span></div>}
      </Card>
    </div>
  )
}
