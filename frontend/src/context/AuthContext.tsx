import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api, type UserProfile } from '../api';

interface AuthContextValue {
  user: UserProfile | null;
  loading: boolean;
  isAdmin: boolean;
  canEdit: boolean;
  isReadOnly: boolean;
  login: (email: string, password: string) => Promise<UserProfile>;
  logout: () => Promise<void>;
  setAuthUser: (user: UserProfile | null) => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshUser = useCallback(async () => {
    try {
      const res = await api.getMe();
      setUser(res?.user ?? null);
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const login = async (email: string, password: string): Promise<UserProfile> => {
    const res = await api.login(email, password);
    setUser(res.user);
    return res.user;
  };

  const logout = async (): Promise<void> => {
    try {
      await api.logout();
    } finally {
      setUser(null);
    }
  };

  const isAdmin = user?.role === 'admin';
  const canEdit = user?.role === 'admin' || user?.role === 'editor';
  const isReadOnly = user?.role === 'viewer';

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        isAdmin,
        canEdit,
        isReadOnly,
        login,
        logout,
        setAuthUser: setUser,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextValue => {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used inside AuthProvider');
  }
  return ctx;
};
