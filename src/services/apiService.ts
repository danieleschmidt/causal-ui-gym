// Real API Service Implementation
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

interface CacheEntry {
  data: any;
  timestamp: number;
  ttl: number;
}

class APIService {
  private client: AxiosInstance;
  private cache: Map<string, CacheEntry>;
  private requestQueue: Map<string, Promise<any>>;
  
  constructor(baseURL: string = process.env.REACT_APP_API_URL || 'http://localhost:3000') {
    this.cache = new Map();
    this.requestQueue = new Map();
    
    this.client = axios.create({
      baseURL,
      timeout: 15000,
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    this.setupInterceptors();
  }
  
  private setupInterceptors() {
    // Request interceptor for auth
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );
    
    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          // Handle token refresh
          const newToken = await this.refreshToken();
          if (newToken) {
            error.config.headers.Authorization = `Bearer ${newToken}`;
            return this.client.request(error.config);
          }
        }
        return Promise.reject(error);
      }
    );
  }
  
  private async refreshToken(): Promise<string | null> {
    try {
      const refreshToken = localStorage.getItem('refresh_token');
      const response = await this.client.post('/auth/refresh', { refreshToken });
      const { accessToken } = response.data;
      localStorage.setItem('auth_token', accessToken);
      return accessToken;
    } catch {
      localStorage.removeItem('auth_token');
      localStorage.removeItem('refresh_token');
      window.location.href = '/login';
      return null;
    }
  }
  
  private getCacheKey(url: string, params?: any): string {
    return `${url}:${JSON.stringify(params || {})}`;
  }
  
  private getFromCache(key: string): any | null {
    const entry = this.cache.get(key);
    if (!entry) return null;
    
    const now = Date.now();
    if (now - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }
    
    return entry.data;
  }
  
  private setCache(key: string, data: any, ttl: number = 60000) {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
  }
  
  async get<T>(url: string, config?: AxiosRequestConfig & { cache?: boolean; ttl?: number }): Promise<T> {
    const cacheKey = this.getCacheKey(url, config?.params);
    
    // Check cache
    if (config?.cache !== false) {
      const cached = this.getFromCache(cacheKey);
      if (cached) return cached;
    }
    
    // Check if request is already in progress
    const existingRequest = this.requestQueue.get(cacheKey);
    if (existingRequest) {
      return existingRequest;
    }
    
    // Make request
    const request = this.client.get<T>(url, config).then(res => {
      const data = res.data;
      this.setCache(cacheKey, data, config?.ttl);
      this.requestQueue.delete(cacheKey);
      return data;
    });
    
    this.requestQueue.set(cacheKey, request);
    return request;
  }
  
  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }
  
  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }
  
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }
  
  // Batch requests
  async batch<T>(requests: Array<{ method: string; url: string; data?: any }>): Promise<T[]> {
    const promises = requests.map(req => {
      switch (req.method.toLowerCase()) {
        case 'get': return this.get(req.url);
        case 'post': return this.post(req.url, req.data);
        case 'put': return this.put(req.url, req.data);
        case 'delete': return this.delete(req.url);
        default: throw new Error(`Unknown method: ${req.method}`);
      }
    });
    
    return Promise.all(promises);
  }
  
  // WebSocket connection
  connectWebSocket(url: string, handlers: {
    onMessage?: (data: any) => void;
    onError?: (error: any) => void;
    onClose?: () => void;
  }) {
    const ws = new WebSocket(url);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handlers.onMessage?.(data);
    };
    
    ws.onerror = handlers.onError || console.error;
    ws.onclose = handlers.onClose || (() => {});
    
    return ws;
  }
}

export default new APIService();
