import React, { createContext, useState, useContext, useEffect } from 'react';
import { authAPI } from '../services/api';
import { storage } from '../utils/storage';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedUser = storage.getUser();
    if (storedUser && storage.isAuthenticated()) {
      setUser(storedUser);
    }
    setLoading(false);
  }, []);

  const login = async (credentials) => {
    try {
      const response = await authAPI.login(credentials);
      const { user, tokens } = response.data;
      
      storage.setTokens(tokens.access, tokens.refresh);
      storage.setUser(user);
      setUser(user);
      
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error || 'Login failed',
      };
    }
  };

  const register = async (userData) => {
    try {
      const response = await authAPI.register(userData);
      const { user, tokens } = response.data;
      
      storage.setTokens(tokens.access, tokens.refresh);
      storage.setUser(user);
      setUser(user);
      
      return { success: true };
    } catch (error) {
      const errorMsg = error.response?.data?.username?.[0] 
        || error.response?.data?.email?.[0]
        || error.response?.data?.password?.[0]
        || 'Registration failed';
      
      return {
        success: false,
        error: errorMsg,
      };
    }
  };

  const logout = async () => {
    try {
      const refreshToken = storage.getRefreshToken();
      if (refreshToken) {
        await authAPI.logout(refreshToken);
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      storage.clearAuth();
      setUser(null);
    }
  };

  const value = {
    user,
    login,
    register,
    logout,
    isAuthenticated: !!user,
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <p>Loading...</p>
      </div>
    );
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};