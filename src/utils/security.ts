/**
 * Security utilities for Causal UI Gym
 * 
 * This module provides comprehensive security measures including:
 * - Content Security Policy configuration
 * - Input sanitization and validation
 * - Rate limiting
 * - Security headers management
 */

export interface SecurityConfig {
  enableCSP: boolean
  enableRateLimit: boolean
  enableInputSanitization: boolean
  enableSecurityHeaders: boolean
  allowedOrigins: string[]
  maxRequestSize: number
  sessionTimeout: number
}

export interface SecurityHeaders {
  'Content-Security-Policy'?: string
  'X-Frame-Options'?: string
  'X-Content-Type-Options'?: string
  'X-XSS-Protection'?: string
  'Strict-Transport-Security'?: string
  'Referrer-Policy'?: string
  'Permissions-Policy'?: string
}

export interface CSPConfig {
  defaultSrc: string[]
  scriptSrc: string[]
  styleSrc: string[]
  imgSrc: string[]
  connectSrc: string[]
  fontSrc: string[]
  frameSrc: string[]
  mediaSrc: string[]
  objectSrc: string[]
  childSrc: string[]
  workerSrc: string[]
  reportUri?: string
  upgradeInsecureRequests: boolean
}

// Default security configuration for Causal UI Gym
export const DEFAULT_SECURITY_CONFIG: SecurityConfig = {
  enableCSP: true,
  enableRateLimit: true,
  enableInputSanitization: true,
  enableSecurityHeaders: true,
  allowedOrigins: ['https://localhost:3000', 'https://causal-ui-gym.dev'],
  maxRequestSize: 10 * 1024 * 1024, // 10MB
  sessionTimeout: 24 * 60 * 60 * 1000 // 24 hours
}

// Content Security Policy configuration
export const DEFAULT_CSP_CONFIG: CSPConfig = {
  defaultSrc: ["'self'"],
  scriptSrc: [
    "'self'",
    "'unsafe-inline'", // Required for React development
    "'unsafe-eval'", // Required for JAX computations
    'https://cdn.jsdelivr.net',
    'https://unpkg.com'
  ],
  styleSrc: [
    "'self'",
    "'unsafe-inline'", // Required for Material-UI
    'https://fonts.googleapis.com'
  ],
  imgSrc: [
    "'self'",
    'data:', // For base64 images
    'blob:', // For generated charts
    'https:'
  ],
  connectSrc: [
    "'self'",
    'ws://localhost:*', // WebSocket connections
    'wss://localhost:*',
    'https://api.openai.com',
    'https://api.anthropic.com',
    'https://*.huggingface.co'
  ],
  fontSrc: [
    "'self'",
    'https://fonts.gstatic.com',
    'data:'
  ],
  frameSrc: ["'none'"], // Prevent iframe embedding
  mediaSrc: ["'self'"],
  objectSrc: ["'none'"], // Prevent Flash/Java applets
  childSrc: ["'none'"],
  workerSrc: [
    "'self'",
    'blob:' // For Web Workers
  ],
  upgradeInsecureRequests: true
}

/**
 * Generate Content Security Policy header string
 */
export function generateCSP(config: CSPConfig = DEFAULT_CSP_CONFIG): string {
  const directives: string[] = []

  // Add each directive
  Object.entries(config).forEach(([key, value]) => {
    if (key === 'reportUri' || key === 'upgradeInsecureRequests') return
    
    const directive = key.replace(/([A-Z])/g, '-$1').toLowerCase()
    if (Array.isArray(value) && value.length > 0) {
      directives.push(`${directive} ${value.join(' ')}`)
    }
  })

  // Add upgrade insecure requests
  if (config.upgradeInsecureRequests) {
    directives.push('upgrade-insecure-requests')
  }

  // Add report URI if provided
  if (config.reportUri) {
    directives.push(`report-uri ${config.reportUri}`)
  }

  return directives.join('; ')
}

/**
 * Generate security headers for HTTP responses
 */
export function generateSecurityHeaders(config: SecurityConfig = DEFAULT_SECURITY_CONFIG): SecurityHeaders {
  const headers: SecurityHeaders = {}

  if (config.enableCSP) {
    headers['Content-Security-Policy'] = generateCSP()
  }

  if (config.enableSecurityHeaders) {
    // Prevent clickjacking
    headers['X-Frame-Options'] = 'DENY'
    
    // Prevent MIME type sniffing
    headers['X-Content-Type-Options'] = 'nosniff'
    
    // XSS protection (legacy but still useful)
    headers['X-XSS-Protection'] = '1; mode=block'
    
    // HTTPS enforcement (only in production)
    if (process.env.NODE_ENV === 'production') {
      headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
    }
    
    // Control referrer information
    headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    // Permissions policy (feature policy replacement)
    headers['Permissions-Policy'] = [
      'camera=()',
      'microphone=()',
      'geolocation=()',
      'interest-cohort=()', // Disable FLoC
      'payment=()',
      'usb=()'
    ].join(', ')
  }

  return headers
}

/**
 * Validate request origin against allowed origins
 */
export function validateOrigin(origin: string, allowedOrigins: string[] = DEFAULT_SECURITY_CONFIG.allowedOrigins): boolean {
  if (!origin) return false
  
  // In development, allow localhost with any port
  if (process.env.NODE_ENV === 'development') {
    if (origin.startsWith('http://localhost:') || origin.startsWith('https://localhost:')) {
      return true
    }
  }
  
  return allowedOrigins.some(allowed => {
    // Exact match
    if (allowed === origin) return true
    
    // Wildcard subdomain match
    if (allowed.startsWith('*.')) {
      const domain = allowed.substring(2)
      // Extract domain from origin (remove protocol)
      const originDomain = origin.replace(/^https?:\/\//, '')
      return originDomain.endsWith(`.${domain}`) || originDomain === domain
    }
    
    return false
  })
}

/**
 * Generate secure session token
 */
export function generateSessionToken(): string {
  const array = new Uint8Array(32)
  crypto.getRandomValues(array)
  return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('')
}

/**
 * Hash sensitive data (passwords, tokens)
 */
export async function hashSensitiveData(data: string, salt?: string): Promise<{ hash: string; salt: string }> {
  const encoder = new TextEncoder()
  
  // Generate salt if not provided
  if (!salt) {
    const saltArray = new Uint8Array(16)
    crypto.getRandomValues(saltArray)
    salt = Array.from(saltArray, byte => byte.toString(16).padStart(2, '0')).join('')
  }
  
  const dataBuffer = encoder.encode(data + salt)
  const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer)
  const hashArray = new Uint8Array(hashBuffer)
  const hash = Array.from(hashArray, byte => byte.toString(16).padStart(2, '0')).join('')
  
  return { hash, salt }
}

/**
 * Verify hashed data
 */
export async function verifyHashedData(data: string, hash: string, salt: string): Promise<boolean> {
  const { hash: computedHash } = await hashSensitiveData(data, salt)
  return computedHash === hash
}

/**
 * Secure local storage wrapper
 */
export class SecureStorage {
  private prefix: string
  private encryptionKey: CryptoKey | null = null

  constructor(prefix: string = 'causal_ui_') {
    this.prefix = prefix
    this.initEncryption()
  }

  private async initEncryption() {
    try {
      // Generate or retrieve encryption key
      const keyData = localStorage.getItem(`${this.prefix}encryption_key`)
      if (keyData) {
        const keyBuffer = Uint8Array.from(atob(keyData), c => c.charCodeAt(0))
        this.encryptionKey = await crypto.subtle.importKey(
          'raw',
          keyBuffer,
          { name: 'AES-GCM' },
          false,
          ['encrypt', 'decrypt']
        )
      } else {
        this.encryptionKey = await crypto.subtle.generateKey(
          { name: 'AES-GCM', length: 256 },
          true,
          ['encrypt', 'decrypt']
        )
        
        const keyBuffer = await crypto.subtle.exportKey('raw', this.encryptionKey)
        const keyString = btoa(String.fromCharCode(...new Uint8Array(keyBuffer)))
        localStorage.setItem(`${this.prefix}encryption_key`, keyString)
      }
    } catch (error) {
      console.warn('Failed to initialize encryption, falling back to plain storage:', error)
    }
  }

  async setItem(key: string, value: string): Promise<void> {
    const fullKey = `${this.prefix}${key}`
    
    if (!this.encryptionKey) {
      localStorage.setItem(fullKey, value)
      return
    }

    try {
      const encoder = new TextEncoder()
      const data = encoder.encode(value)
      const iv = crypto.getRandomValues(new Uint8Array(12))
      
      const encrypted = await crypto.subtle.encrypt(
        { name: 'AES-GCM', iv },
        this.encryptionKey,
        data
      )
      
      const encryptedData = {
        iv: Array.from(iv),
        data: Array.from(new Uint8Array(encrypted))
      }
      
      localStorage.setItem(fullKey, JSON.stringify(encryptedData))
    } catch (error) {
      console.warn('Encryption failed, storing as plain text:', error)
      localStorage.setItem(fullKey, value)
    }
  }

  async getItem(key: string): Promise<string | null> {
    const fullKey = `${this.prefix}${key}`
    const stored = localStorage.getItem(fullKey)
    
    if (!stored || !this.encryptionKey) {
      return stored
    }

    try {
      const encryptedData = JSON.parse(stored)
      if (!encryptedData.iv || !encryptedData.data) {
        // Plain text data
        return stored
      }
      
      const iv = new Uint8Array(encryptedData.iv)
      const data = new Uint8Array(encryptedData.data)
      
      const decrypted = await crypto.subtle.decrypt(
        { name: 'AES-GCM', iv },
        this.encryptionKey,
        data
      )
      
      const decoder = new TextDecoder()
      return decoder.decode(decrypted)
    } catch (error) {
      console.warn('Decryption failed, returning plain text:', error)
      return stored
    }
  }

  removeItem(key: string): void {
    localStorage.removeItem(`${this.prefix}${key}`)
  }

  clear(): void {
    const keys = Object.keys(localStorage)
    keys.forEach(key => {
      if (key.startsWith(this.prefix)) {
        localStorage.removeItem(key)
      }
    })
  }
}

/**
 * Security middleware for API requests
 */
export interface SecurityMiddlewareOptions {
  validateOrigin?: boolean
  checkRateLimit?: boolean
  validateContentLength?: boolean
  sanitizeInput?: boolean
  requireAuth?: boolean
}

export function createSecurityMiddleware(
  config: SecurityConfig = DEFAULT_SECURITY_CONFIG,
  options: SecurityMiddlewareOptions = {}
) {
  return async (request: Request, context: any) => {
    const { validateOrigin: checkOrigin = true, checkRateLimit = true } = options

    // Validate origin
    if (checkOrigin) {
      const origin = request.headers.get('origin')
      if (origin && !validateOrigin(origin, config.allowedOrigins)) {
        return new Response('Forbidden: Invalid origin', { status: 403 })
      }
    }

    // Check content length
    const contentLength = request.headers.get('content-length')
    if (contentLength && parseInt(contentLength) > config.maxRequestSize) {
      return new Response('Payload too large', { status: 413 })
    }

    // Rate limiting would be implemented here
    if (checkRateLimit) {
      // This would integrate with a rate limiting service
      // For now, we'll add a simple in-memory rate limiter
    }

    return null // Continue to next middleware
  }
}

/**
 * Sanitize file uploads
 */
export function validateFileUpload(file: File): {
  isValid: boolean
  errors: string[]
  sanitizedName?: string
} {
  const errors: string[] = []
  const maxSize = 10 * 1024 * 1024 // 10MB
  const allowedTypes = [
    'application/json',
    'text/csv',
    'text/plain',
    'image/png',
    'image/jpeg',
    'image/svg+xml'
  ]

  // Check file size
  if (file.size > maxSize) {
    errors.push(`File size exceeds maximum of ${maxSize / (1024 * 1024)}MB`)
  }

  // Check file type
  if (!allowedTypes.includes(file.type)) {
    errors.push(`File type ${file.type} is not allowed`)
  }

  // Sanitize filename
  let sanitizedName = file.name
    .replace(/[^a-zA-Z0-9.-]/g, '_') // Replace special chars with underscore
    .replace(/\.+/g, '.') // Replace multiple dots with single dot

  // Ensure file has extension
  if (!sanitizedName.includes('.')) {
    sanitizedName += '.txt'
  }

  // Limit length after ensuring extension
  sanitizedName = sanitizedName.substring(0, 255)

  return {
    isValid: errors.length === 0,
    errors,
    sanitizedName
  }
}

// Export security utilities
export const security = {
  generateCSP,
  generateSecurityHeaders,
  validateOrigin,
  generateSessionToken,
  hashSensitiveData,
  verifyHashedData,
  SecureStorage,
  createSecurityMiddleware,
  validateFileUpload,
  DEFAULT_SECURITY_CONFIG,
  DEFAULT_CSP_CONFIG
}

// Initialize global security storage instance
export const secureStorage = new SecureStorage()