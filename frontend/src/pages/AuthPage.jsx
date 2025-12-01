import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Shield, Mail, Lock, User, CheckCircle, AlertCircle } from 'lucide-react';
import { authStyles as styles } from '../styles/auth.styles';

const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    password2: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { login, register } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (!isLogin && formData.password !== formData.password2) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    try {
      let result;
      if (isLogin) {
        result = await login({
          username: formData.username,
          password: formData.password,
        });
      } else {
        result = await register({
          username: formData.username,
          email: formData.email,
          password: formData.password,
          password2: formData.password2,
        });
      }

      if (result.success) {
        navigate('/dashboard');
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      {/* Left Panel - Branding */}
      <div style={styles.leftPanel}>
        <div style={styles.leftPanelOverlay}></div>
        <div style={styles.brandSection}>
          <div style={styles.logoIcon}>
            <Shield size={40} color="white" />
          </div>
          <h1 style={styles.brandTitle}>Content Moderation AI</h1>
          <p style={styles.brandSubtitle}>
            Enterprise-grade content moderation powered by advanced AI and policy automation
          </p>
          
          <div style={styles.featureList}>
            <div style={styles.featureItem}>
              <div style={styles.featureIcon}>
                <CheckCircle size={16} />
              </div>
              <div style={styles.featureText}>
                Automated policy compliance checking with AI-powered analysis
              </div>
            </div>
            <div style={styles.featureItem}>
              <div style={styles.featureIcon}>
                <CheckCircle size={16} />
              </div>
              <div style={styles.featureText}>
                Real-time violation detection and review management
              </div>
            </div>
            <div style={styles.featureItem}>
              <div style={styles.featureIcon}>
                <CheckCircle size={16} />
              </div>
              <div style={styles.featureText}>
                Comprehensive audit trails and detailed reporting
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Form */}
      <div style={styles.rightPanel}>
        <div style={styles.formContainer}>
          <div style={styles.formHeader}>
            <h2 style={styles.formTitle}>
              {isLogin ? 'Welcome back' : 'Create your account'}
            </h2>
            <p style={styles.formSubtitle}>
              {isLogin 
                ? 'Enter your credentials to access your dashboard' 
                : 'Get started with your free account today'}
            </p>
          </div>

          {error && (
            <div style={styles.alert}>
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} style={styles.form}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Username</label>
              <div style={styles.inputWrapper}>
                <User size={18} style={styles.inputIcon} />
                <input
                  type="text"
                  name="username"
                  value={formData.username}
                  onChange={handleChange}
                  required
                  style={styles.input}
                  placeholder="Enter your username"
                />
              </div>
            </div>

            {!isLogin && (
              <div style={styles.formGroup}>
                <label style={styles.label}>Email address</label>
                <div style={styles.inputWrapper}>
                  <Mail size={18} style={styles.inputIcon} />
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    required
                    style={styles.input}
                    placeholder="Enter your email"
                  />
                </div>
              </div>
            )}

            <div style={styles.formGroup}>
              <label style={styles.label}>Password</label>
              <div style={styles.inputWrapper}>
                <Lock size={18} style={styles.inputIcon} />
                <input
                  type="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  required
                  style={styles.input}
                  placeholder="Enter your password"
                />
              </div>
            </div>

            {!isLogin && (
              <div style={styles.formGroup}>
                <label style={styles.label}>Confirm password</label>
                <div style={styles.inputWrapper}>
                  <Lock size={18} style={styles.inputIcon} />
                  <input
                    type="password"
                    name="password2"
                    value={formData.password2}
                    onChange={handleChange}
                    required
                    style={styles.input}
                    placeholder="Confirm your password"
                  />
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              style={{
                ...styles.submitButton,
                opacity: loading ? 0.7 : 1,
                cursor: loading ? 'not-allowed' : 'pointer',
              }}
            >
              {loading ? 'Please wait...' : (isLogin ? 'Sign in' : 'Create account')}
            </button>
          </form>

          <div style={styles.toggleSection}>
            <span style={styles.toggleText}>
              {isLogin ? "Don't have an account?" : 'Already have an account?'}
            </span>
            <button
              onClick={() => {
                setIsLogin(!isLogin);
                setError('');
                setFormData({ username: '', email: '', password: '', password2: '' });
              }}
              style={styles.toggleButton}
            >
              {isLogin ? 'Sign up' : 'Sign in'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;