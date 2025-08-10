import { describe, it, expect, beforeEach, vi } from 'vitest'
import { 
  generateCSP, 
  generateSecurityHeaders, 
  validateOrigin,
  generateSessionToken,
  hashSensitiveData,
  verifyHashedData,
  SecureStorage,
  validateFileUpload,
  DEFAULT_SECURITY_CONFIG,
  DEFAULT_CSP_CONFIG
} from './security'

// Mock crypto for testing
Object.defineProperty(global, 'crypto', {
  value: {
    getRandomValues: vi.fn((array: Uint8Array) => {
      for (let i = 0; i < array.length; i++) {
        array[i] = Math.floor(Math.random() * 256)
      }
      return array
    }),
    subtle: {
      digest: vi.fn((algorithm: string, data: ArrayBuffer) => {
        // Simple mock hash - just return the input length as a pattern
        const hash = new ArrayBuffer(32)
        const view = new Uint8Array(hash)
        for (let i = 0; i < 32; i++) {
          view[i] = (data.byteLength + i) % 256
        }
        return Promise.resolve(hash)
      }),
      generateKey: vi.fn(() => Promise.resolve({})),
      exportKey: vi.fn(() => Promise.resolve(new ArrayBuffer(32))),
      importKey: vi.fn(() => Promise.resolve({})),
      encrypt: vi.fn(() => Promise.resolve(new ArrayBuffer(16))),
      decrypt: vi.fn(() => Promise.resolve(new ArrayBuffer(16)))
    }
  } as any
})

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  key: vi.fn(),
  length: 0
}
Object.defineProperty(global, 'localStorage', {
  value: localStorageMock
})

describe('Security Utils', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('generateCSP', () => {
    it('should generate valid CSP header', () => {
      const csp = generateCSP()
      expect(csp).toContain("default-src 'self'")
      expect(csp).toContain("script-src")
      expect(csp).toContain("style-src")
      expect(csp).toContain('upgrade-insecure-requests')
    })

    it('should handle custom CSP config', () => {
      const config = {
        ...DEFAULT_CSP_CONFIG,
        scriptSrc: ["'self'", 'https://trusted.com'],
        reportUri: 'https://example.com/csp-report'
      }
      
      const csp = generateCSP(config)
      expect(csp).toContain("script-src 'self' https://trusted.com")
      expect(csp).toContain('report-uri https://example.com/csp-report')
    })

    it('should exclude empty directive arrays', () => {
      const config = {
        ...DEFAULT_CSP_CONFIG,
        frameSrc: []
      }
      
      const csp = generateCSP(config)
      expect(csp).not.toContain('frame-src')
    })
  })

  describe('generateSecurityHeaders', () => {
    it('should generate security headers with CSP enabled', () => {
      const headers = generateSecurityHeaders({ ...DEFAULT_SECURITY_CONFIG, enableCSP: true })
      
      expect(headers).toHaveProperty('Content-Security-Policy')
      expect(headers).toHaveProperty('X-Frame-Options', 'DENY')
      expect(headers).toHaveProperty('X-Content-Type-Options', 'nosniff')
      expect(headers).toHaveProperty('X-XSS-Protection', '1; mode=block')
    })

    it('should exclude CSP when disabled', () => {
      const headers = generateSecurityHeaders({ ...DEFAULT_SECURITY_CONFIG, enableCSP: false })
      
      expect(headers).not.toHaveProperty('Content-Security-Policy')
    })

    it('should include HSTS in production', () => {
      const originalEnv = process.env.NODE_ENV
      process.env.NODE_ENV = 'production'
      
      const headers = generateSecurityHeaders(DEFAULT_SECURITY_CONFIG)
      
      expect(headers).toHaveProperty('Strict-Transport-Security')
      
      process.env.NODE_ENV = originalEnv
    })

    it('should exclude security headers when disabled', () => {
      const headers = generateSecurityHeaders({ 
        ...DEFAULT_SECURITY_CONFIG, 
        enableSecurityHeaders: false 
      })
      
      expect(headers).not.toHaveProperty('X-Frame-Options')
      expect(headers).not.toHaveProperty('X-Content-Type-Options')
    })
  })

  describe('validateOrigin', () => {
    it('should allow exact origin matches', () => {
      expect(validateOrigin('https://localhost:3000')).toBe(true)
      expect(validateOrigin('https://causal-ui-gym.dev')).toBe(true)
    })

    it('should reject unlisted origins', () => {
      expect(validateOrigin('https://malicious.com')).toBe(false)
      expect(validateOrigin('http://evil.org')).toBe(false)
    })

    it('should handle wildcard subdomains', () => {
      const allowedOrigins = ['*.example.com', 'https://app.test.com']
      
      expect(validateOrigin('https://sub.example.com', allowedOrigins)).toBe(true)
      expect(validateOrigin('https://deep.sub.example.com', allowedOrigins)).toBe(true)
      expect(validateOrigin('https://example.com', allowedOrigins)).toBe(true)
      expect(validateOrigin('https://notexample.com', allowedOrigins)).toBe(false)
    })

    it('should allow localhost in development', () => {
      const originalEnv = process.env.NODE_ENV
      process.env.NODE_ENV = 'development'
      
      expect(validateOrigin('http://localhost:8080')).toBe(true)
      expect(validateOrigin('https://localhost:3000')).toBe(true)
      
      process.env.NODE_ENV = originalEnv
    })

    it('should reject empty origins', () => {
      expect(validateOrigin('')).toBe(false)
      expect(validateOrigin(null as any)).toBe(false)
    })
  })

  describe('generateSessionToken', () => {
    it('should generate a session token', () => {
      const token = generateSessionToken()
      
      expect(typeof token).toBe('string')
      expect(token.length).toBe(64) // 32 bytes * 2 hex chars
      expect(/^[a-f0-9]{64}$/.test(token)).toBe(true)
    })

    it('should generate different tokens each time', () => {
      const token1 = generateSessionToken()
      const token2 = generateSessionToken()
      
      expect(token1).not.toBe(token2)
    })
  })

  describe('hashSensitiveData', () => {
    it('should hash data with generated salt', async () => {
      const result = await hashSensitiveData('password123')
      
      expect(result).toHaveProperty('hash')
      expect(result).toHaveProperty('salt')
      expect(typeof result.hash).toBe('string')
      expect(typeof result.salt).toBe('string')
      expect(result.hash.length).toBe(64) // SHA-256 hex string
      expect(result.salt.length).toBe(32) // 16 bytes * 2 hex chars
    })

    it('should use provided salt', async () => {
      const salt = 'fixed-salt-for-testing'
      const result = await hashSensitiveData('password123', salt)
      
      expect(result.salt).toBe(salt)
    })

    it('should produce consistent hashes with same salt', async () => {
      const salt = 'consistent-salt'
      const result1 = await hashSensitiveData('password123', salt)
      const result2 = await hashSensitiveData('password123', salt)
      
      expect(result1.hash).toBe(result2.hash)
      expect(result1.salt).toBe(result2.salt)
    })
  })

  describe('verifyHashedData', () => {
    it('should verify correct password', async () => {
      const { hash, salt } = await hashSensitiveData('password123')
      const isValid = await verifyHashedData('password123', hash, salt)
      
      expect(isValid).toBe(true)
    })

    it('should reject incorrect password', async () => {
      const { hash, salt } = await hashSensitiveData('password123')
      const isValid = await verifyHashedData('wrongpassword', hash, salt)
      
      expect(isValid).toBe(false)
    })
  })

  describe('SecureStorage', () => {
    let storage: SecureStorage

    beforeEach(() => {
      storage = new SecureStorage('test_')
      localStorageMock.getItem.mockReturnValue(null)
    })

    it('should store and retrieve items', async () => {
      // Mock encryption failure so it falls back to plain text
      vi.mocked(crypto.subtle.encrypt).mockRejectedValue(new Error('Mock encryption failure'))
      
      await storage.setItem('key1', 'value1')
      
      expect(localStorageMock.setItem).toHaveBeenCalledWith('test_key1', 'value1')
      
      // Mock the retrieval
      localStorageMock.getItem.mockReturnValue('value1')
      
      const retrieved = await storage.getItem('key1')
      expect(retrieved).toBe('value1')
    })

    it('should remove items', () => {
      storage.removeItem('key1')
      
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('test_key1')
    })

    it('should clear all items with prefix', () => {
      // Mock some keys in localStorage
      const mockKeys = ['test_key1', 'test_key2', 'other_key']
      Object.defineProperty(localStorageMock, 'length', { value: mockKeys.length })
      localStorageMock.key.mockImplementation((index: number) => mockKeys[index])
      
      // Mock Object.keys to return our mock keys
      const originalKeys = Object.keys
      Object.keys = vi.fn().mockReturnValue(mockKeys)
      
      storage.clear()
      
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('test_key1')
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('test_key2')
      expect(localStorageMock.removeItem).not.toHaveBeenCalledWith('other_key')
      
      // Restore Object.keys
      Object.keys = originalKeys
    })
  })

  describe('validateFileUpload', () => {
    const createMockFile = (name: string, size: number, type: string): File => {
      return {
        name,
        size,
        type,
        lastModified: Date.now(),
        webkitRelativePath: ''
      } as File
    }

    it('should accept valid files', () => {
      const file = createMockFile('data.json', 1024, 'application/json')
      const result = validateFileUpload(file)
      
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
      expect(result.sanitizedName).toBe('data.json')
    })

    it('should reject oversized files', () => {
      const file = createMockFile('large.json', 20 * 1024 * 1024, 'application/json') // 20MB
      const result = validateFileUpload(file)
      
      expect(result.isValid).toBe(false)
      expect(result.errors.some(e => e.includes('File size exceeds'))).toBe(true)
    })

    it('should reject invalid file types', () => {
      const file = createMockFile('script.exe', 1024, 'application/x-executable')
      const result = validateFileUpload(file)
      
      expect(result.isValid).toBe(false)
      expect(result.errors.some(e => e.includes('File type'))).toBe(true)
    })

    it('should sanitize filenames', () => {
      const file = createMockFile('file with spaces & special chars!.json', 1024, 'application/json')
      const result = validateFileUpload(file)
      
      expect(result.sanitizedName).toBe('file_with_spaces___special_chars_.json')
    })

    it('should add extension if missing', () => {
      const file = createMockFile('noextension', 1024, 'application/json')
      const result = validateFileUpload(file)
      
      expect(result.sanitizedName).toBe('noextension.txt')
    })

    it('should limit filename length', () => {
      const longName = 'a'.repeat(300) + '.json'
      const file = createMockFile(longName, 1024, 'application/json')
      const result = validateFileUpload(file)
      
      expect(result.sanitizedName!.length).toBeLessThanOrEqual(255)
    })
  })
})