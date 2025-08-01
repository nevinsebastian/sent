import axios from 'axios';

const API_BASE_URL = 'http://localhost:5001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface User {
  id: number;
  email: string;
  name: string;
}

export interface Feedback {
  id: number;
  user_id: number;
  text: string;
  emotion: string;
}

export interface SignupData {
  email: string;
  password: string;
  name: string;
}

export interface LoginData {
  email: string;
  password: string;
}

export interface FeedbackData {
  user_id: number;
  text: string;
}

export const authAPI = {
  signup: async (data: SignupData) => {
    const response = await api.post('/signup', data);
    return response.data;
  },

  login: async (data: LoginData) => {
    const response = await api.post('/login', data);
    return response.data;
  },

  adminLogin: async (data: LoginData) => {
    const response = await api.post('/admin/login', data);
    return response.data;
  },
};

export const feedbackAPI = {
  submit: async (data: FeedbackData) => {
    const response = await api.post('/feedback', data);
    return response.data;
  },

  getAll: async () => {
    const response = await api.get('/admin/feedbacks');
    return response.data;
  },

  analyzeEmotion: async (text: string) => {
    const response = await api.post('/emotion', { text });
    return response.data;
  },
};

export default api; 