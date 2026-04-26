import axios from 'axios';

// Use environment variable for API URL in production, fallback to proxy in development
const API_BASE_URL = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Simple in-memory cache for GET requests
const cache = new Map();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

const getCachedData = (key) => {
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    return cached.data;
  }
  cache.delete(key);
  return null;
};

const setCachedData = (key, data) => {
  cache.set(key, { data, timestamp: Date.now() });
};

export const fetchHealth = () => api.get('/health');

export const fetchClusters = () => {
  const cacheKey = 'clusters';
  const cached = getCachedData(cacheKey);
  if (cached) return Promise.resolve({ data: cached });
  
  return api.get('/clusters').then(response => {
    setCachedData(cacheKey, response.data);
    return response;
  });
};

export const fetchClusterDetail = (id) => api.get(`/clusters/${id}`);
export const predictSeverity = (data) => api.post('/predict', data);

export const fetchEda = (type) => {
  const cacheKey = `eda-${type}`;
  const cached = getCachedData(cacheKey);
  if (cached) return Promise.resolve({ data: cached });
  
  return api.get(`/eda/${type}`).then(response => {
    setCachedData(cacheKey, response.data);
    return response;
  });
};

export const fetchModelMetrics = () => {
  const cacheKey = 'model-metrics';
  const cached = getCachedData(cacheKey);
  if (cached) return Promise.resolve({ data: cached });
  
  return api.get('/model/metrics').then(response => {
    setCachedData(cacheKey, response.data);
    return response;
  });
};

export const fetchEdaSummary = () => {
  const cacheKey = 'eda-summary';
  const cached = getCachedData(cacheKey);
  if (cached) return Promise.resolve({ data: cached });
  
  return api.get('/eda/summary').then(response => {
    setCachedData(cacheKey, response.data);
    return response;
  });
};

export const uploadCSV = (file) => {
  const fd = new FormData();
  fd.append('file', file);
  return api.post('/upload', fd);
};

export const runPipeline = () => {
  // Clear cache when pipeline runs
  cache.clear();
  return api.post('/pipeline/run');
};

export const fetchUploads = () => api.get('/uploads');
export const deleteUpload = (id, clearArtifacts = false) =>
  api.delete(`/uploads/${id}${clearArtifacts ? '?clear_artifacts=true' : ''}`);

export default api;
