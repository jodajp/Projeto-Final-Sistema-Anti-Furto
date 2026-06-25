/**
 * Centralized API Configuration
 * Dynamically resolves the API host port based on current page address,
 * avoiding hardcoded VM IPs during local testing and production.
 */
export const getApiBaseUrl = () => {
  const hostname = window.location.hostname;
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://127.0.0.1:8000';
  }
  // Deployed environment: port 8000 on the same domain name or IP
  return `http://${hostname}:8000`;
};

export const API_BASE_URL = getApiBaseUrl();
export const API_BASE = `${API_BASE_URL}/api`;
export const API_URL = `${API_BASE}/metricas/cluster`;
