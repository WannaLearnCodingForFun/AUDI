import { useCallback, useEffect, useRef, useState } from 'react'
import { errorText, wsUrl } from './api'

/** Run an async loader; returns { data, error, loading, reload }. */
export function useAsync(loader, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  const seq = useRef(0)
  const run = useCallback(() => {
    const id = ++seq.current
    setState((s) => ({ ...s, loading: true }))
    return loader()
      .then((data) => id === seq.current && setState({ data, error: null, loading: false }))
      .catch((e) => id === seq.current && setState({ data: null, error: errorText(e), loading: false }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  useEffect(() => { run() }, [run])
  return { ...state, reload: run }
}

/** Authenticated WebSocket with automatic reconnect. `onMessage` receives parsed JSON. */
export function useSocket(path, onMessage, enabled = true, reconnectMs = 3000) {
  const ref = useRef(null)
  const handler = useRef(onMessage)
  handler.current = onMessage
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    if (!enabled) return undefined
    let closed = false
    let timer = null
    const open = () => {
      const ws = new WebSocket(wsUrl(path))
      ws.binaryType = 'arraybuffer'
      ref.current = ws
      ws.onopen = () => setConnected(true)
      ws.onmessage = (e) => {
        if (typeof e.data === 'string') handler.current(JSON.parse(e.data))
      }
      ws.onclose = () => {
        setConnected(false)
        if (!closed) timer = setTimeout(open, reconnectMs)
      }
    }
    open()
    return () => {
      closed = true
      clearTimeout(timer)
      ref.current?.close()
    }
  }, [path, enabled, reconnectMs])

  const send = useCallback((payload) => {
    const ws = ref.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(payload)
      return true
    }
    return false
  }, [])

  return { connected, send, bufferedAmount: () => ref.current?.bufferedAmount ?? 0 }
}
