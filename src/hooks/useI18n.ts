/**
 * Internationalization hook for the Causal UI Gym
 * 
 * Provides translation capabilities with regional formatting,
 * accessibility support, and compliance features.
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { translations, getNestedTranslation, defaultLanguage, type TranslationDict } from '../i18n/translations'

export interface I18nConfig {
  locale: string
  region?: string
  timezone?: string
  currency?: string
  dateFormat?: string
  numberFormat?: string
  rtl?: boolean
  fallback?: string
}

export interface UseI18nReturn {
  t: (key: string, params?: Record<string, string | number>) => string
  locale: string
  region: string
  timezone: string
  currency: string
  dateFormat: string
  numberFormat: string
  rtl: boolean
  changeLocale: (newLocale: string) => Promise<void>
  formatDate: (date: Date, options?: Intl.DateTimeFormatOptions) => string
  formatNumber: (number: number, options?: Intl.NumberFormatOptions) => string
  formatCurrency: (amount: number, currency?: string) => string
  formatRelativeTime: (date: Date) => string
  getLocaleInfo: () => LocaleInfo
  isLocaleSupported: (locale: string) => boolean
  getBrowserLocale: () => string
  getRegionalPreferences: () => RegionalPreferences
}

export interface LocaleInfo {
  locale: string
  region: string
  language: string
  script?: string
  nativeName: string
  englishName: string
  direction: 'ltr' | 'rtl'
  calendar: string
  firstDayOfWeek: number
}

export interface RegionalPreferences {
  timezone: string
  currency: string
  dateFormat: string
  timeFormat: string
  numberFormat: string
  measurementSystem: 'metric' | 'imperial'
  temperatureUnit: 'celsius' | 'fahrenheit'
}

// Locale information database
const localeInfo: Record<string, LocaleInfo> = {
  en: {
    locale: 'en',
    region: 'US',
    language: 'en',
    nativeName: 'English',
    englishName: 'English',
    direction: 'ltr',
    calendar: 'gregory',
    firstDayOfWeek: 0
  },
  'en-GB': {
    locale: 'en-GB',
    region: 'GB',
    language: 'en',
    nativeName: 'English (UK)',
    englishName: 'English (United Kingdom)',
    direction: 'ltr',
    calendar: 'gregory',
    firstDayOfWeek: 1
  },
  es: {
    locale: 'es',
    region: 'ES',
    language: 'es',
    nativeName: 'Español',
    englishName: 'Spanish',
    direction: 'ltr',
    calendar: 'gregory',
    firstDayOfWeek: 1
  },
  fr: {
    locale: 'fr',
    region: 'FR',
    language: 'fr',
    nativeName: 'Français',
    englishName: 'French',
    direction: 'ltr',
    calendar: 'gregory',
    firstDayOfWeek: 1
  },
  de: {
    locale: 'de',
    region: 'DE',
    language: 'de',
    nativeName: 'Deutsch',
    englishName: 'German',
    direction: 'ltr',
    calendar: 'gregory',
    firstDayOfWeek: 1
  },
  ja: {
    locale: 'ja',
    region: 'JP',
    language: 'ja',
    nativeName: '日本語',
    englishName: 'Japanese',
    direction: 'ltr',
    calendar: 'gregory',
    firstDayOfWeek: 0
  },
  zh: {
    locale: 'zh',
    region: 'CN',
    language: 'zh',
    script: 'Hans',
    nativeName: '中文（简体）',
    englishName: 'Chinese (Simplified)',
    direction: 'ltr',
    calendar: 'gregory',
    firstDayOfWeek: 1
  },
  ar: {
    locale: 'ar',
    region: 'SA',
    language: 'ar',
    script: 'Arab',
    nativeName: 'العربية',
    englishName: 'Arabic',
    direction: 'rtl',
    calendar: 'islamic',
    firstDayOfWeek: 6
  }
}

// Regional preferences by country/region
const regionalPreferences: Record<string, RegionalPreferences> = {
  US: {
    timezone: 'America/New_York',
    currency: 'USD',
    dateFormat: 'MM/dd/yyyy',
    timeFormat: '12h',
    numberFormat: '1,234.56',
    measurementSystem: 'imperial',
    temperatureUnit: 'fahrenheit'
  },
  GB: {
    timezone: 'Europe/London',
    currency: 'GBP',
    dateFormat: 'dd/MM/yyyy',
    timeFormat: '24h',
    numberFormat: '1,234.56',
    measurementSystem: 'metric',
    temperatureUnit: 'celsius'
  },
  DE: {
    timezone: 'Europe/Berlin',
    currency: 'EUR',
    dateFormat: 'dd.MM.yyyy',
    timeFormat: '24h',
    numberFormat: '1.234,56',
    measurementSystem: 'metric',
    temperatureUnit: 'celsius'
  },
  FR: {
    timezone: 'Europe/Paris',
    currency: 'EUR',
    dateFormat: 'dd/MM/yyyy',
    timeFormat: '24h',
    numberFormat: '1 234,56',
    measurementSystem: 'metric',
    temperatureUnit: 'celsius'
  },
  JP: {
    timezone: 'Asia/Tokyo',
    currency: 'JPY',
    dateFormat: 'yyyy/MM/dd',
    timeFormat: '24h',
    numberFormat: '1,234',
    measurementSystem: 'metric',
    temperatureUnit: 'celsius'
  },
  CN: {
    timezone: 'Asia/Shanghai',
    currency: 'CNY',
    dateFormat: 'yyyy/MM/dd',
    timeFormat: '24h',
    numberFormat: '1,234.56',
    measurementSystem: 'metric',
    temperatureUnit: 'celsius'
  },
  ES: {
    timezone: 'Europe/Madrid',
    currency: 'EUR',
    dateFormat: 'dd/MM/yyyy',
    timeFormat: '24h',
    numberFormat: '1.234,56',
    measurementSystem: 'metric',
    temperatureUnit: 'celsius'
  }
}

// Storage keys
const LOCALE_STORAGE_KEY = 'causal-ui-locale'
const I18N_CONFIG_STORAGE_KEY = 'causal-ui-i18n-config'

export function useI18n(initialConfig?: Partial<I18nConfig>): UseI18nReturn {
  const [config, setConfig] = useState<I18nConfig>(() => {
    // Load from localStorage or use defaults
    const stored = localStorage.getItem(I18N_CONFIG_STORAGE_KEY)
    const browserLocale = getBrowserLocale()
    
    const defaultConfig: I18nConfig = {
      locale: browserLocale,
      region: getRegionFromLocale(browserLocale),
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      currency: 'USD',
      dateFormat: 'MM/dd/yyyy',
      numberFormat: '1,234.56',
      rtl: false,
      fallback: defaultLanguage,
      ...initialConfig
    }
    
    if (stored) {
      try {
        const parsedConfig = JSON.parse(stored)
        return { ...defaultConfig, ...parsedConfig }
      } catch (error) {
        console.warn('Failed to parse stored i18n config:', error)
      }
    }
    
    return defaultConfig
  })

  // Save config to localStorage
  useEffect(() => {
    localStorage.setItem(I18N_CONFIG_STORAGE_KEY, JSON.stringify(config))
    localStorage.setItem(LOCALE_STORAGE_KEY, config.locale)
    
    // Set document language and direction
    document.documentElement.lang = config.locale
    document.documentElement.dir = config.rtl ? 'rtl' : 'ltr'
    
    // Set CSS custom properties for internationalization
    document.documentElement.style.setProperty('--text-direction', config.rtl ? 'rtl' : 'ltr')
    document.documentElement.style.setProperty('--start-direction', config.rtl ? 'right' : 'left')
    document.documentElement.style.setProperty('--end-direction', config.rtl ? 'left' : 'right')
    
  }, [config])

  // Translation function with parameter interpolation
  const t = useCallback((key: string, params?: Record<string, string | number>): string => {
    const locale = config.locale.split('-')[0] // Get base language
    const translation = translations[locale as keyof typeof translations] as TranslationDict
    
    if (!translation) {
      console.warn(`Translation not found for locale: ${locale}`)
      return key
    }
    
    let translatedText = getNestedTranslation(translation, key)
    
    // If translation not found, try fallback locale
    if (translatedText === key && config.fallback && config.fallback !== locale) {
      const fallbackTranslation = translations[config.fallback as keyof typeof translations] as TranslationDict
      if (fallbackTranslation) {
        translatedText = getNestedTranslation(fallbackTranslation, key)
      }
    }
    
    // Parameter interpolation
    if (params && typeof translatedText === 'string') {
      Object.entries(params).forEach(([paramKey, value]) => {
        const placeholder = `{{${paramKey}}}`
        translatedText = translatedText.replace(new RegExp(placeholder, 'g'), String(value))
      })
    }
    
    return translatedText
  }, [config.locale, config.fallback])

  // Change locale with validation and async loading
  const changeLocale = useCallback(async (newLocale: string): Promise<void> => {
    if (!isLocaleSupported(newLocale)) {
      console.warn(`Locale ${newLocale} is not supported`)
      return
    }
    
    try {
      // Update configuration
      const info = getLocaleInfo(newLocale)
      const regional = getRegionalPreferences(info.region)
      
      setConfig(prev => ({
        ...prev,
        locale: newLocale,
        region: info.region,
        timezone: regional.timezone,
        currency: regional.currency,
        dateFormat: regional.dateFormat,
        numberFormat: regional.numberFormat,
        rtl: info.direction === 'rtl'
      }))
      
      // Notify screen readers of language change
      const announcement = document.createElement('div')
      announcement.setAttribute('aria-live', 'polite')
      announcement.setAttribute('aria-atomic', 'true')
      announcement.className = 'sr-only'
      announcement.textContent = `Language changed to ${info.englishName}`
      document.body.appendChild(announcement)
      
      setTimeout(() => {
        document.body.removeChild(announcement)
      }, 1000)
      
    } catch (error) {
      console.error('Failed to change locale:', error)
      throw error
    }
  }, [])

  // Date formatting with locale support
  const formatDate = useCallback((date: Date, options?: Intl.DateTimeFormatOptions): string => {
    const defaultOptions: Intl.DateTimeFormatOptions = {
      timeZone: config.timezone,
      ...options
    }
    
    try {
      return new Intl.DateTimeFormat(config.locale, defaultOptions).format(date)
    } catch (error) {
      console.warn('Date formatting failed:', error)
      return date.toLocaleDateString(config.fallback, defaultOptions)
    }
  }, [config.locale, config.timezone, config.fallback])

  // Number formatting with locale support
  const formatNumber = useCallback((number: number, options?: Intl.NumberFormatOptions): string => {
    try {
      return new Intl.NumberFormat(config.locale, options).format(number)
    } catch (error) {
      console.warn('Number formatting failed:', error)
      return new Intl.NumberFormat(config.fallback, options).format(number)
    }
  }, [config.locale, config.fallback])

  // Currency formatting
  const formatCurrency = useCallback((amount: number, currency?: string): string => {
    const currencyCode = currency || config.currency
    
    try {
      return new Intl.NumberFormat(config.locale, {
        style: 'currency',
        currency: currencyCode
      }).format(amount)
    } catch (error) {
      console.warn('Currency formatting failed:', error)
      return `${currencyCode} ${amount.toFixed(2)}`
    }
  }, [config.locale, config.currency])

  // Relative time formatting
  const formatRelativeTime = useCallback((date: Date): string => {
    const now = new Date()
    const diffInSeconds = (date.getTime() - now.getTime()) / 1000
    
    const units: Array<[string, number]> = [
      ['year', 365 * 24 * 60 * 60],
      ['month', 30 * 24 * 60 * 60],
      ['day', 24 * 60 * 60],
      ['hour', 60 * 60],
      ['minute', 60],
      ['second', 1]
    ]
    
    for (const [unit, secondsInUnit] of units) {
      const amount = Math.abs(diffInSeconds) / secondsInUnit
      if (amount >= 1) {
        try {
          const rtf = new Intl.RelativeTimeFormat(config.locale, { numeric: 'auto' })
          return rtf.format(
            Math.round(diffInSeconds / secondsInUnit),
            unit as Intl.RelativeTimeFormatUnit
          )
        } catch (error) {
          console.warn('Relative time formatting failed:', error)
          return date.toLocaleDateString(config.locale)
        }
      }
    }
    
    return t('common.now', {})
  }, [config.locale, t])

  // Memoized computed values
  const computedValues = useMemo(() => ({
    locale: config.locale,
    region: config.region || 'US',
    timezone: config.timezone || 'UTC',
    currency: config.currency || 'USD',
    dateFormat: config.dateFormat || 'MM/dd/yyyy',
    numberFormat: config.numberFormat || '1,234.56',
    rtl: config.rtl || false
  }), [config])

  return {
    t,
    ...computedValues,
    changeLocale,
    formatDate,
    formatNumber,
    formatCurrency,
    formatRelativeTime,
    getLocaleInfo: () => getLocaleInfo(config.locale),
    isLocaleSupported,
    getBrowserLocale,
    getRegionalPreferences: () => getRegionalPreferences(config.region || 'US')
  }
}

// Helper functions
function getBrowserLocale(): string {
  if (typeof navigator !== 'undefined') {
    return navigator.language || navigator.languages?.[0] || defaultLanguage
  }
  return defaultLanguage
}

function getRegionFromLocale(locale: string): string {
  const parts = locale.split('-')
  if (parts.length > 1) {
    return parts[1].toUpperCase()
  }
  
  // Default regions for base languages
  const defaultRegions: Record<string, string> = {
    en: 'US',
    es: 'ES',
    fr: 'FR',
    de: 'DE',
    ja: 'JP',
    zh: 'CN',
    ar: 'SA'
  }
  
  return defaultRegions[parts[0]] || 'US'
}

function isLocaleSupported(locale: string): boolean {
  const baseLocale = locale.split('-')[0]
  return Object.keys(translations).includes(baseLocale)
}

function getLocaleInfo(locale: string): LocaleInfo {
  return localeInfo[locale] || localeInfo[locale.split('-')[0]] || localeInfo.en
}

function getRegionalPreferences(region: string): RegionalPreferences {
  return regionalPreferences[region] || regionalPreferences.US
}

// Export for external use
export {
  getBrowserLocale,
  isLocaleSupported,
  getLocaleInfo,
  getRegionalPreferences,
  localeInfo,
  regionalPreferences
}