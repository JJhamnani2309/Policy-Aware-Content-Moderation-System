import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests if available
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('accessToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle token refresh on 401
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem('refreshToken');
        const response = await axios.post(`${API_URL}/api/auth/refresh/`, {
          refresh: refreshToken,
        });

        const { access } = response.data;
        localStorage.setItem('accessToken', access);

        originalRequest.headers.Authorization = `Bearer ${access}`;
        return api(originalRequest);
      } catch (refreshError) {
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
        localStorage.removeItem('user');
        window.location.href = '/';
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  register: (userData) => api.post('/api/auth/register/', userData),
  login: (credentials) => api.post('/api/auth/login/', credentials),
  logout: (refreshToken) => api.post('/api/auth/logout/', { refresh_token: refreshToken }),
  getProfile: () => api.get('/api/auth/profile/'),
};

// Policy API
export const policyAPI = {
  uploadPolicies: (files) => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    return api.post('/api/moderation/upload-policy/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  listPolicies: () => api.get('/api/moderation/policies/'),
  deletePolicy: (id) => api.delete(`/api/moderation/policies/${id}/`),
  clearPolicies: () => api.post('/api/moderation/clear-policies/'),
};

// Moderation API
export const moderationAPI = {
  moderateFile: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/api/moderation/moderate/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  getHistory: () => api.get('/api/moderation/history/'),
  getDetail: (id) => api.get(`/api/moderation/history/${id}/`),
  updateFinalVerdict: (id, verdict) => api.post(`/api/moderation/history/${id}/verdict/`, {
    final_verdict: verdict,
  }),
};

export default api;