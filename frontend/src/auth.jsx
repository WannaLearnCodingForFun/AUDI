import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { getToken, login as apiLogin, me, setToken, setUnauthorizedHandler } from './api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [ready, setReady] = useState(false)

  const logout = useCallback(() => {
    setToken(null)
    setUser(null)
  }, [])

  useEffect(() => {
    setUnauthorizedHandler(logout)
    if (!getToken()) {
      setReady(true)
      return
    }
    me().then(setUser).catch(logout).finally(() => setReady(true))
  }, [logout])

  const login = async (username, password) => {
    const { access_token: token } = await apiLogin(username, password)
    setToken(token)
    const u = await me()
    setUser(u)
    return u
  }

  return <AuthContext.Provider value={{ user, ready, login, logout }}>{children}</AuthContext.Provider>
}

export const useAuth = () => useContext(AuthContext)

export const hasRole = (user, ...roles) => !!user && (user.role === 'admin' || roles.includes(user.role))
