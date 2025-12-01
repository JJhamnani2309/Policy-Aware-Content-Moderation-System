import React from 'react';
import { useAuth } from '../context/AuthContext';
import { LogOut, Shield } from 'lucide-react';
import { navbarStyles as styles } from '../styles/navbar.styles';

const Navbar = () => {
  const { user, logout } = useAuth();

  return (
    <nav style={styles.navbar}>
      <div style={styles.container}>
        <div style={styles.brand}>
          <div style={styles.iconWrapper}>
            <Shield size={28} style={styles.icon} />
          </div>
          <div style={styles.brandText}>
            <div style={styles.brandTitle}>Content Moderation AI</div>
            <div style={styles.brandSubtitle}>AI-Powered Policy Enforcement</div>
          </div>
        </div>
        
        <div style={styles.rightSection}>
          <div style={styles.userInfo}>
            <div style={styles.userName}>{user?.username || 'User'}</div>
            <div style={styles.userRole}>Administrator</div>
          </div>
          <button onClick={logout} style={styles.logoutBtn}>
            <LogOut size={18} />
            <span style={{ marginLeft: '8px' }}>Sign Out</span>
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;