/**
 * Accessibility Utilities
 * 
 * Provides comprehensive accessibility support including WCAG compliance,
 * screen reader compatibility, keyboard navigation, and inclusive design features.
 */

export interface AccessibilityPreferences {
  prefersReducedMotion: boolean
  prefersHighContrast: boolean
  prefersColorScheme: 'light' | 'dark' | 'auto'
  fontSize: 'small' | 'medium' | 'large' | 'extra-large'
  screenReaderEnabled: boolean
  keyboardNavigation: boolean
  focusIndicators: boolean
  captionsEnabled: boolean
  audioDescriptions: boolean
}

export interface FocusManagementConfig {
  trapFocus: boolean
  restoreFocus: boolean
  skipLinks: boolean
  focusRingVisible: boolean
  customFocusRing?: string
}

export interface AriaAnnouncement {
  message: string
  priority: 'polite' | 'assertive'
  timeout?: number
}

export interface ColorContrastResult {
  ratio: number
  level: 'AA' | 'AAA' | 'fail'
  isLargeText: boolean
  passes: boolean
}

export interface KeyboardNavigationMap {
  [key: string]: {
    action: string
    description: string
    scope?: 'global' | 'component' | 'modal'
  }
}

// WCAG Color Contrast Thresholds
const CONTRAST_THRESHOLDS = {
  AA_NORMAL: 4.5,
  AA_LARGE: 3,
  AAA_NORMAL: 7,
  AAA_LARGE: 4.5
}

// Default keyboard navigation shortcuts
export const DEFAULT_KEYBOARD_MAP: KeyboardNavigationMap = {
  'Tab': {
    action: 'focus_next',
    description: 'Move to next focusable element',
    scope: 'global'
  },
  'Shift+Tab': {
    action: 'focus_previous', 
    description: 'Move to previous focusable element',
    scope: 'global'
  },
  'Enter': {
    action: 'activate',
    description: 'Activate focused element',
    scope: 'global'
  },
  'Space': {
    action: 'activate',
    description: 'Activate button or toggle',
    scope: 'component'
  },
  'Escape': {
    action: 'close',
    description: 'Close modal or cancel operation',
    scope: 'modal'
  },
  'ArrowUp': {
    action: 'navigate_up',
    description: 'Navigate up in lists or decrease value',
    scope: 'component'
  },
  'ArrowDown': {
    action: 'navigate_down',
    description: 'Navigate down in lists or increase value',
    scope: 'component'
  },
  'ArrowLeft': {
    action: 'navigate_left',
    description: 'Navigate left or decrease value',
    scope: 'component'
  },
  'ArrowRight': {
    action: 'navigate_right',
    description: 'Navigate right or increase value',
    scope: 'component'
  },
  'Home': {
    action: 'go_first',
    description: 'Go to first item',
    scope: 'component'
  },
  'End': {
    action: 'go_last',
    description: 'Go to last item',
    scope: 'component'
  },
  'Alt+1': {
    action: 'skip_to_main',
    description: 'Skip to main content',
    scope: 'global'
  },
  'Alt+2': {
    action: 'skip_to_nav',
    description: 'Skip to navigation',
    scope: 'global'
  }
}

// Screen reader only styles
export const SCREEN_READER_ONLY_STYLES = {
  position: 'absolute',
  width: '1px',
  height: '1px',
  padding: '0',
  margin: '-1px',
  overflow: 'hidden',
  clip: 'rect(0, 0, 0, 0)',
  whiteSpace: 'nowrap',
  border: '0'
} as const

/**
 * Get user accessibility preferences from system and browser
 */
export function getAccessibilityPreferences(): AccessibilityPreferences {
  const mediaQueries = {
    prefersReducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    prefersHighContrast: window.matchMedia('(prefers-contrast: high)').matches,
    prefersColorScheme: window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }

  return {
    prefersReducedMotion: mediaQueries.prefersReducedMotion,
    prefersHighContrast: mediaQueries.prefersHighContrast,
    prefersColorScheme: mediaQueries.prefersColorScheme as 'light' | 'dark',
    fontSize: 'medium',
    screenReaderEnabled: detectScreenReader(),
    keyboardNavigation: true,
    focusIndicators: true,
    captionsEnabled: false,
    audioDescriptions: false
  }
}

/**
 * Detect if a screen reader is likely being used
 */
export function detectScreenReader(): boolean {
  // Check for common screen reader user agents
  const userAgent = navigator.userAgent.toLowerCase()
  const screenReaderUA = [
    'nvda', 'jaws', 'windoweyes', 'voiceover', 'talkback', 'dragon'
  ]
  
  // Check for reduced motion preference (often used by screen reader users)
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  
  // Check for speech synthesis API usage
  const speechSynthesis = 'speechSynthesis' in window
  
  return screenReaderUA.some(sr => userAgent.includes(sr)) || 
         prefersReducedMotion || 
         (speechSynthesis && speechSynthesis)
}

/**
 * Calculate color contrast ratio between two colors
 */
export function calculateColorContrast(color1: string, color2: string): ColorContrastResult {
  const luminance1 = getLuminance(color1)
  const luminance2 = getLuminance(color2)
  
  const lighter = Math.max(luminance1, luminance2)
  const darker = Math.min(luminance1, luminance2)
  
  const ratio = (lighter + 0.05) / (darker + 0.05)
  
  return {
    ratio,
    level: getContrastLevel(ratio),
    isLargeText: false, // Will be determined by font size in implementation
    passes: ratio >= CONTRAST_THRESHOLDS.AA_NORMAL
  }
}

/**
 * Get relative luminance of a color
 */
function getLuminance(color: string): number {
  const rgb = hexToRgb(color)
  if (!rgb) return 0
  
  const [r, g, b] = [rgb.r, rgb.g, rgb.b].map(c => {
    c = c / 255
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
  })
  
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

/**
 * Convert hex color to RGB
 */
function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null
}

/**
 * Determine WCAG contrast level
 */
function getContrastLevel(ratio: number): 'AA' | 'AAA' | 'fail' {
  if (ratio >= CONTRAST_THRESHOLDS.AAA_NORMAL) return 'AAA'
  if (ratio >= CONTRAST_THRESHOLDS.AA_NORMAL) return 'AA'
  return 'fail'
}

/**
 * Create and manage ARIA live regions for announcements
 */
export class AriaLiveRegion {
  private politeRegion: HTMLElement | null = null
  private assertiveRegion: HTMLElement | null = null

  constructor() {
    this.createRegions()
  }

  private createRegions(): void {
    // Create polite live region
    this.politeRegion = document.createElement('div')
    this.politeRegion.setAttribute('aria-live', 'polite')
    this.politeRegion.setAttribute('aria-atomic', 'true')
    this.politeRegion.setAttribute('aria-relevant', 'additions text')
    this.politeRegion.className = 'sr-only'
    Object.assign(this.politeRegion.style, SCREEN_READER_ONLY_STYLES)
    
    // Create assertive live region
    this.assertiveRegion = document.createElement('div')
    this.assertiveRegion.setAttribute('aria-live', 'assertive')
    this.assertiveRegion.setAttribute('aria-atomic', 'true')
    this.assertiveRegion.setAttribute('aria-relevant', 'additions text')
    this.assertiveRegion.className = 'sr-only'
    Object.assign(this.assertiveRegion.style, SCREEN_READER_ONLY_STYLES)
    
    document.body.appendChild(this.politeRegion)
    document.body.appendChild(this.assertiveRegion)
  }

  /**
   * Announce a message to screen readers
   */
  announce(announcement: AriaAnnouncement): void {
    const region = announcement.priority === 'assertive' ? this.assertiveRegion : this.politeRegion
    
    if (region) {
      // Clear existing content
      region.textContent = ''
      
      // Add new message after a brief delay to ensure it's announced
      setTimeout(() => {
        region.textContent = announcement.message
      }, 10)
      
      // Clear message after timeout
      if (announcement.timeout) {
        setTimeout(() => {
          region.textContent = ''
        }, announcement.timeout)
      }
    }
  }

  /**
   * Clean up live regions
   */
  destroy(): void {
    if (this.politeRegion) {
      document.body.removeChild(this.politeRegion)
      this.politeRegion = null
    }
    if (this.assertiveRegion) {
      document.body.removeChild(this.assertiveRegion)
      this.assertiveRegion = null
    }
  }
}

/**
 * Focus management utilities
 */
export class FocusManager {
  private focusStack: HTMLElement[] = []
  private trapContainer: HTMLElement | null = null
  private lastFocusedElement: HTMLElement | null = null

  /**
   * Set focus trap within a container
   */
  trapFocus(container: HTMLElement): void {
    this.trapContainer = container
    this.lastFocusedElement = document.activeElement as HTMLElement
    
    const focusableElements = this.getFocusableElements(container)
    if (focusableElements.length === 0) return
    
    // Focus first element
    focusableElements[0].focus()
    
    // Add event listener for tab cycling
    container.addEventListener('keydown', this.handleTrapKeydown.bind(this))
  }

  /**
   * Remove focus trap
   */
  removeFocusTrap(): void {
    if (this.trapContainer) {
      this.trapContainer.removeEventListener('keydown', this.handleTrapKeydown.bind(this))
      this.trapContainer = null
    }
    
    // Restore focus to last focused element
    if (this.lastFocusedElement) {
      this.lastFocusedElement.focus()
      this.lastFocusedElement = null
    }
  }

  /**
   * Handle keyboard navigation within focus trap
   */
  private handleTrapKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Tab' || !this.trapContainer) return
    
    const focusableElements = this.getFocusableElements(this.trapContainer)
    if (focusableElements.length === 0) return
    
    const firstElement = focusableElements[0]
    const lastElement = focusableElements[focusableElements.length - 1]
    
    if (event.shiftKey) {
      // Shift + Tab
      if (document.activeElement === firstElement) {
        event.preventDefault()
        lastElement.focus()
      }
    } else {
      // Tab
      if (document.activeElement === lastElement) {
        event.preventDefault()
        firstElement.focus()
      }
    }
  }

  /**
   * Get all focusable elements within a container
   */
  private getFocusableElements(container: HTMLElement): HTMLElement[] {
    const focusableSelectors = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable="true"]'
    ]
    
    const elements = container.querySelectorAll(focusableSelectors.join(','))
    return Array.from(elements) as HTMLElement[]
  }

  /**
   * Save current focus for later restoration
   */
  saveFocus(): void {
    this.focusStack.push(document.activeElement as HTMLElement)
  }

  /**
   * Restore previously saved focus
   */
  restoreFocus(): void {
    const element = this.focusStack.pop()
    if (element && element.focus) {
      element.focus()
    }
  }
}

/**
 * Keyboard navigation handler
 */
export class KeyboardNavigationHandler {
  private keyMap: KeyboardNavigationMap
  private handlers: Map<string, (event: KeyboardEvent) => void> = new Map()

  constructor(keyMap: KeyboardNavigationMap = DEFAULT_KEYBOARD_MAP) {
    this.keyMap = keyMap
    this.setupGlobalListeners()
  }

  /**
   * Set up global keyboard listeners
   */
  private setupGlobalListeners(): void {
    document.addEventListener('keydown', this.handleGlobalKeydown.bind(this))
  }

  /**
   * Handle global keyboard events
   */
  private handleGlobalKeydown(event: KeyboardEvent): void {
    const key = this.getKeyCombo(event)
    const mapping = this.keyMap[key]
    
    if (mapping && mapping.scope === 'global') {
      const handler = this.handlers.get(mapping.action)
      if (handler) {
        event.preventDefault()
        handler(event)
      }
    }
  }

  /**
   * Get key combination string
   */
  private getKeyCombo(event: KeyboardEvent): string {
    const modifiers = []
    if (event.ctrlKey) modifiers.push('Ctrl')
    if (event.altKey) modifiers.push('Alt')
    if (event.shiftKey) modifiers.push('Shift')
    if (event.metaKey) modifiers.push('Meta')
    
    modifiers.push(event.key)
    return modifiers.join('+')
  }

  /**
   * Register a handler for a keyboard action
   */
  registerHandler(action: string, handler: (event: KeyboardEvent) => void): void {
    this.handlers.set(action, handler)
  }

  /**
   * Unregister a handler
   */
  unregisterHandler(action: string): void {
    this.handlers.delete(action)
  }

  /**
   * Get help text for keyboard shortcuts
   */
  getKeyboardHelp(scope?: string): string[] {
    return Object.entries(this.keyMap)
      .filter(([_, mapping]) => !scope || mapping.scope === scope)
      .map(([key, mapping]) => `${key}: ${mapping.description}`)
  }
}

/**
 * Skip link utilities
 */
export function createSkipLinks(): void {
  const skipLinksContainer = document.createElement('div')
  skipLinksContainer.className = 'skip-links'
  skipLinksContainer.setAttribute('aria-label', 'Skip navigation links')
  
  const skipLinks = [
    { href: '#main-content', text: 'Skip to main content' },
    { href: '#navigation', text: 'Skip to navigation' },
    { href: '#search', text: 'Skip to search' }
  ]
  
  skipLinks.forEach(link => {
    const skipLink = document.createElement('a')
    skipLink.href = link.href
    skipLink.textContent = link.text
    skipLink.className = 'skip-link'
    
    // Style skip link (hidden by default, visible on focus)
    Object.assign(skipLink.style, {
      position: 'absolute',
      top: '-40px',
      left: '6px',
      background: '#000',
      color: '#fff',
      padding: '8px',
      textDecoration: 'none',
      borderRadius: '4px',
      zIndex: '100000',
      transition: 'top 0.3s'
    })
    
    skipLink.addEventListener('focus', () => {
      skipLink.style.top = '6px'
    })
    
    skipLink.addEventListener('blur', () => {
      skipLink.style.top = '-40px'
    })
    
    skipLinksContainer.appendChild(skipLink)
  })
  
  document.body.insertBefore(skipLinksContainer, document.body.firstChild)
}

/**
 * Generate accessible IDs for form controls
 */
export function generateAccessibleId(prefix: string = 'accessible'): string {
  return `${prefix}-${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Create accessible error message for form fields
 */
export function createAccessibleErrorMessage(fieldId: string, message: string): HTMLElement {
  const errorElement = document.createElement('div')
  errorElement.id = `${fieldId}-error`
  errorElement.setAttribute('role', 'alert')
  errorElement.setAttribute('aria-live', 'polite')
  errorElement.className = 'error-message'
  errorElement.textContent = message
  
  return errorElement
}

/**
 * Validate WCAG compliance for an element
 */
export function validateWCAGCompliance(element: HTMLElement): {
  passes: boolean
  violations: string[]
  warnings: string[]
} {
  const violations: string[] = []
  const warnings: string[] = []
  
  // Check for missing alt text on images
  const images = element.querySelectorAll('img')
  images.forEach(img => {
    if (!img.getAttribute('alt') && !img.hasAttribute('aria-hidden')) {
      violations.push(`Image missing alt text: ${img.src}`)
    }
  })
  
  // Check for missing labels on form controls
  const formControls = element.querySelectorAll('input, select, textarea')
  formControls.forEach(control => {
    const hasLabel = document.querySelector(`label[for="${control.id}"]`) ||
                    control.hasAttribute('aria-label') ||
                    control.hasAttribute('aria-labelledby')
    
    if (!hasLabel) {
      violations.push(`Form control missing label: ${control.tagName}`)
    }
  })
  
  // Check for missing heading hierarchy
  const headings = element.querySelectorAll('h1, h2, h3, h4, h5, h6')
  let lastLevel = 0
  headings.forEach(heading => {
    const level = parseInt(heading.tagName.substr(1))
    if (level > lastLevel + 1) {
      warnings.push(`Heading level skipped: ${heading.tagName} after h${lastLevel}`)
    }
    lastLevel = level
  })
  
  // Check for keyboard accessibility
  const interactiveElements = element.querySelectorAll('button, a, input, select, textarea, [tabindex]')
  interactiveElements.forEach(el => {
    const tabIndex = el.getAttribute('tabindex')
    if (tabIndex && parseInt(tabIndex) > 0) {
      warnings.push(`Positive tabindex detected: ${el.tagName}`)
    }
  })
  
  return {
    passes: violations.length === 0,
    violations,
    warnings
  }
}

/**
 * Apply accessibility preferences to the document
 */
export function applyAccessibilityPreferences(preferences: AccessibilityPreferences): void {
  const root = document.documentElement
  
  // Apply reduced motion
  if (preferences.prefersReducedMotion) {
    root.style.setProperty('--motion-duration', '0s')
    root.style.setProperty('--motion-distance', '0px')
  }
  
  // Apply high contrast
  if (preferences.prefersHighContrast) {
    root.classList.add('high-contrast')
  }
  
  // Apply color scheme
  root.setAttribute('data-color-scheme', preferences.prefersColorScheme)
  
  // Apply font size
  const fontSizeMap = {
    'small': '0.875rem',
    'medium': '1rem',
    'large': '1.125rem',
    'extra-large': '1.25rem'
  }
  root.style.setProperty('--base-font-size', fontSizeMap[preferences.fontSize])
  
  // Apply focus indicators
  if (preferences.focusIndicators) {
    root.classList.add('focus-indicators-enabled')
  }
}

// Export instances for global use
export const ariaLiveRegion = new AriaLiveRegion()
export const focusManager = new FocusManager()
export const keyboardHandler = new KeyboardNavigationHandler()

// Initialize skip links on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', createSkipLinks)
} else {
  createSkipLinks()
}