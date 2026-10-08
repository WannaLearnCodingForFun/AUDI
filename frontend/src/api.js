import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const getHealth = () => api.get('/health').then(r => r.data)
export const getSummary = () => api.get('/summary').then(r => r.data)
export const getAssets = () => api.get('/assets').then(r => r.data)
export const getAsset = (id) => api.get(`/assets/${id}`).then(r => r.data)
export const getAssetHistory = (id) => api.get(`/assets/${id}/history`).then(r => r.data)
export const searchMemory = (q) => api.get('/memory/search', { params: { q } }).then(r => r.data)
export const analyzeImage = (file, addToMemory = true) => {
  const form = new FormData()
  form.append('file', file)
  return api.post(`/analyze?add_to_memory=${addToMemory}`, form).then(r => r.data)
}
export const simulateMission = () => api.post('/simulate-mission').then(r => r.data)
export const getMissions = () => api.get('/missions').then(r => r.data)

export default api
