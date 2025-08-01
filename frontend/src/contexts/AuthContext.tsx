import React, { createContext, useContext, useState, ReactNode } from 'react';

interface AuthContextType {
  user: { id: number; email: string; name: string; isAdmin?: boolean } | null;
  isAdmin: boolean;
  login: (userData: { id: number; email: string; name: string; isAdmin?: boolean }) => void;
  logout: () => void;
  adminLogin: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<{ id: number; email: string; name: string } | null>(null);
  const [isAdmin, setIsAdmin] = useState(false);

  const login = (userData: { id: number; email: string; name: string; isAdmin?: boolean }) => {
    setUser(userData);
    setIsAdmin(userData.isAdmin || false);
    localStorage.setItem('user', JSON.stringify(userData));
    if (userData.isAdmin) {
      localStorage.setItem('isAdmin', 'true');
    }
  };

  const logout = () => {
    setUser(null);
    setIsAdmin(false);
    localStorage.removeItem('user');
    localStorage.removeItem('isAdmin');
  };

  const adminLogin = () => {
    const adminUser = { id: 0, email: 'admin', name: 'Administrator', isAdmin: true };
    setUser(adminUser);
    setIsAdmin(true);
    localStorage.setItem('user', JSON.stringify(adminUser));
    localStorage.setItem('isAdmin', 'true');
  };

  return (
    <AuthContext.Provider value={{ user, isAdmin, login, logout, adminLogin }}>
      {children}
    </AuthContext.Provider>
  );
}; 