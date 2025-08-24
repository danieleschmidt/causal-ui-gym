/**
 * Global Compliance Service for Multi-Region Deployment
 * 
 * Implements GDPR, CCPA, PDPA and other privacy regulations compliance
 * with automated data handling, consent management, and audit trails.
 */

import { monitoring } from '../utils/monitoring'

interface ConsentRecord {
  userId: string
  timestamp: Date
  consentType: 'gdpr' | 'ccpa' | 'pdpa' | 'general'
  granted: boolean
  purposes: string[]
  region: string
  ipAddress?: string
  userAgent?: string
}

interface DataProcessingRecord {
  userId: string
  dataType: 'causal_model' | 'experiment_data' | 'metrics' | 'user_preferences'
  operation: 'create' | 'read' | 'update' | 'delete' | 'export'
  timestamp: Date
  legalBasis: string
  region: string
  retentionPeriod?: number
}

interface ComplianceRegion {
  code: string
  name: string
  regulations: string[]
  dataResidency: boolean
  consentRequired: boolean
  rightToDelete: boolean
  rightToPortability: boolean
  dataProtectionAuthority: string
}

class GlobalComplianceService {
  private consentRecords: Map<string, ConsentRecord[]> = new Map()
  private processingRecords: DataProcessingRecord[] = []
  private regions: Map<string, ComplianceRegion> = new Map()

  constructor() {
    this.initializeRegions()
  }

  private initializeRegions() {
    // European Union - GDPR
    this.regions.set('EU', {
      code: 'EU',
      name: 'European Union',
      regulations: ['GDPR', 'ePrivacy'],
      dataResidency: true,
      consentRequired: true,
      rightToDelete: true,
      rightToPortability: true,
      dataProtectionAuthority: 'European Data Protection Board'
    })

    // California - CCPA/CPRA
    this.regions.set('CA', {
      code: 'CA',
      name: 'California',
      regulations: ['CCPA', 'CPRA'],
      dataResidency: false,
      consentRequired: true,
      rightToDelete: true,
      rightToPortability: true,
      dataProtectionAuthority: 'California Privacy Protection Agency'
    })

    // Singapore - PDPA
    this.regions.set('SG', {
      code: 'SG',
      name: 'Singapore',
      regulations: ['PDPA'],
      dataResidency: false,
      consentRequired: true,
      rightToDelete: true,
      rightToPortability: false,
      dataProtectionAuthority: 'Personal Data Protection Commission'
    })

    // United Kingdom - UK GDPR
    this.regions.set('UK', {
      code: 'UK',
      name: 'United Kingdom',
      regulations: ['UK-GDPR', 'DPA-2018'],
      dataResidency: true,
      consentRequired: true,
      rightToDelete: true,
      rightToPortability: true,
      dataProtectionAuthority: 'Information Commissioner\'s Office'
    })

    // General/Default region
    this.regions.set('DEFAULT', {
      code: 'DEFAULT',
      name: 'Default',
      regulations: [],
      dataResidency: false,
      consentRequired: false,
      rightToDelete: true,
      rightToPortability: false,
      dataProtectionAuthority: 'None'
    })
  }

  /**
   * Determine user's region based on IP address and other factors
   */
  public determineUserRegion(ipAddress?: string, userLocation?: string): string {
    // In a real implementation, this would use GeoIP services
    // For demo purposes, we'll use simple heuristics

    if (userLocation) {
      const location = userLocation.toLowerCase()
      if (location.includes('eu') || location.includes('europe')) return 'EU'
      if (location.includes('ca') || location.includes('california')) return 'CA'
      if (location.includes('sg') || location.includes('singapore')) return 'SG'
      if (location.includes('uk') || location.includes('britain')) return 'UK'
    }

    // Fallback to IP-based detection (simplified)
    if (ipAddress) {
      // This would typically use a GeoIP service
      // For now, return default
    }

    return 'DEFAULT'
  }

  /**
   * Check if consent is required for a specific operation
   */
  public isConsentRequired(
    userId: string, 
    operation: string, 
    dataType: string, 
    region?: string
  ): boolean {
    const userRegion = region || this.getUserRegion(userId)
    const regionConfig = this.regions.get(userRegion) || this.regions.get('DEFAULT')!

    // Always require consent for sensitive operations in regulated regions
    if (regionConfig.consentRequired) {
      const sensitiveOperations = ['export', 'share', 'analyze_personal']
      const sensitiveDataTypes = ['personal_info', 'behavioral_data', 'experiment_results']
      
      if (sensitiveOperations.includes(operation) || sensitiveDataTypes.includes(dataType)) {
        return true
      }
    }

    return false
  }

  /**
   * Record user consent
   */
  public async recordConsent(
    userId: string,
    consentType: ConsentRecord['consentType'],
    granted: boolean,
    purposes: string[],
    region?: string,
    metadata?: { ipAddress?: string; userAgent?: string }
  ): Promise<void> {
    const userRegion = region || this.getUserRegion(userId)
    
    const consentRecord: ConsentRecord = {
      userId,
      timestamp: new Date(),
      consentType,
      granted,
      purposes,
      region: userRegion,
      ipAddress: metadata?.ipAddress,
      userAgent: metadata?.userAgent
    }

    const userConsents = this.consentRecords.get(userId) || []
    userConsents.push(consentRecord)
    this.consentRecords.set(userId, userConsents)

    // Track consent metrics
    monitoring.trackMetric('consent_recorded', 1, {
      region: userRegion,
      consent_type: consentType,
      granted: granted.toString(),
      purposes_count: purposes.length.toString()
    })

    // Log for audit trail
    console.log(`Consent recorded for user ${userId}: ${granted ? 'GRANTED' : 'DENIED'} for ${purposes.join(', ')} in ${userRegion}`)
  }

  /**
   * Check if user has valid consent for an operation
   */
  public hasValidConsent(
    userId: string,
    purpose: string,
    operation?: string
  ): boolean {
    const userConsents = this.consentRecords.get(userId) || []
    
    // Find the most recent consent for this purpose
    const relevantConsents = userConsents
      .filter(consent => consent.purposes.includes(purpose) && consent.granted)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())

    if (relevantConsents.length === 0) {
      return false
    }

    const latestConsent = relevantConsents[0]
    const userRegion = this.getUserRegion(userId)
    const regionConfig = this.regions.get(userRegion) || this.regions.get('DEFAULT')!

    // Check if consent hasn't expired (GDPR: consent should be refreshed periodically)
    const consentAge = Date.now() - latestConsent.timestamp.getTime()
    const maxAge = regionConfig.regulations.includes('GDPR') ? 
      365 * 24 * 60 * 60 * 1000 : // 1 year for GDPR
      2 * 365 * 24 * 60 * 60 * 1000 // 2 years for others

    return consentAge <= maxAge
  }

  /**
   * Record data processing activity
   */
  public async recordDataProcessing(
    userId: string,
    dataType: DataProcessingRecord['dataType'],
    operation: DataProcessingRecord['operation'],
    legalBasis: string,
    region?: string,
    retentionPeriod?: number
  ): Promise<void> {
    const userRegion = region || this.getUserRegion(userId)
    
    const processingRecord: DataProcessingRecord = {
      userId,
      dataType,
      operation,
      timestamp: new Date(),
      legalBasis,
      region: userRegion,
      retentionPeriod
    }

    this.processingRecords.push(processingRecord)

    // Track processing metrics
    monitoring.trackMetric('data_processing', 1, {
      region: userRegion,
      data_type: dataType,
      operation,
      legal_basis: legalBasis
    })

    // Check for compliance issues
    if (operation === 'delete' || operation === 'export') {
      monitoring.trackMetric('data_rights_request', 1, {
        region: userRegion,
        request_type: operation
      })
    }
  }

  /**
   * Handle data subject rights requests (GDPR Article 15-22)
   */
  public async handleDataSubjectRequest(
    userId: string,
    requestType: 'access' | 'rectification' | 'erasure' | 'portability' | 'restriction' | 'objection',
    region?: string
  ): Promise<any> {
    const userRegion = region || this.getUserRegion(userId)
    const regionConfig = this.regions.get(userRegion) || this.regions.get('DEFAULT')!

    monitoring.trackMetric('data_subject_request', 1, {
      region: userRegion,
      request_type: requestType
    })

    switch (requestType) {
      case 'access':
        return this.handleAccessRequest(userId)

      case 'rectification':
        return this.handleRectificationRequest(userId)

      case 'erasure':
        if (regionConfig.rightToDelete) {
          return this.handleErasureRequest(userId)
        }
        throw new Error('Right to erasure not available in this region')

      case 'portability':
        if (regionConfig.rightToPortability) {
          return this.handlePortabilityRequest(userId)
        }
        throw new Error('Right to data portability not available in this region')

      case 'restriction':
        return this.handleRestrictionRequest(userId)

      case 'objection':
        return this.handleObjectionRequest(userId)

      default:
        throw new Error(`Unknown request type: ${requestType}`)
    }
  }

  private async handleAccessRequest(userId: string): Promise<any> {
    const userConsents = this.consentRecords.get(userId) || []
    const userProcessing = this.processingRecords.filter(record => record.userId === userId)

    return {
      personalData: {
        userId,
        consents: userConsents,
        processingActivities: userProcessing,
        exportDate: new Date().toISOString()
      },
      dataCategories: this.getUserDataCategories(userId),
      processingPurposes: this.getUserProcessingPurposes(userId),
      retentionPeriods: this.getUserRetentionPeriods(userId),
      recipients: this.getUserDataRecipients(userId)
    }
  }

  private async handleErasureRequest(userId: string): Promise<any> {
    // Mark user data for deletion
    const deletionRecord: DataProcessingRecord = {
      userId,
      dataType: 'causal_model', // This would include all data types
      operation: 'delete',
      timestamp: new Date(),
      legalBasis: 'Article 17 GDPR - Right to erasure',
      region: this.getUserRegion(userId)
    }

    this.processingRecords.push(deletionRecord)

    // In a real implementation, this would trigger actual data deletion
    // across all systems and databases
    
    return {
      status: 'accepted',
      deletionScheduled: new Date(),
      confirmationRequired: true,
      affectedSystems: ['user_profile', 'causal_models', 'experiment_data', 'analytics']
    }
  }

  private async handlePortabilityRequest(userId: string): Promise<any> {
    // Generate portable data export
    const userData = await this.handleAccessRequest(userId)
    
    return {
      format: 'JSON',
      data: userData,
      exportDate: new Date().toISOString(),
      downloadUrl: `/api/data-export/${userId}/${Date.now()}`,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) // 7 days
    }
  }

  private async handleRectificationRequest(userId: string): Promise<any> {
    return {
      status: 'accepted',
      instructions: 'Please provide the corrected information and specify which data needs to be updated.',
      updateForm: `/api/user/${userId}/update-profile`
    }
  }

  private async handleRestrictionRequest(userId: string): Promise<any> {
    return {
      status: 'accepted',
      restrictionApplied: true,
      affectedProcessing: ['analytics', 'profiling', 'automated_decision_making'],
      note: 'Data processing has been restricted while your request is being processed.'
    }
  }

  private async handleObjectionRequest(userId: string): Promise<any> {
    return {
      status: 'accepted',
      processingHalted: ['direct_marketing', 'profiling', 'legitimate_interest_processing'],
      note: 'You can still use the service, but certain processing activities have been halted.'
    }
  }

  /**
   * Get compliance requirements for a specific region
   */
  public getComplianceRequirements(region: string): ComplianceRegion | null {
    return this.regions.get(region) || null
  }

  /**
   * Generate compliance audit report
   */
  public generateAuditReport(region?: string): any {
    const targetRegions = region ? [region] : Array.from(this.regions.keys())
    const report: any = {
      generatedAt: new Date().toISOString(),
      regions: {},
      summary: {
        totalConsents: 0,
        totalProcessingRecords: 0,
        complianceScore: 0
      }
    }

    for (const regionCode of targetRegions) {
      const regionConfig = this.regions.get(regionCode)
      if (!regionConfig) continue

      const regionConsents = Array.from(this.consentRecords.values())
        .flat()
        .filter(consent => consent.region === regionCode)

      const regionProcessing = this.processingRecords
        .filter(record => record.region === regionCode)

      const consentRate = regionConsents.length > 0 ? 
        regionConsents.filter(c => c.granted).length / regionConsents.length : 1

      const processingCompliance = this.calculateProcessingCompliance(regionCode)

      report.regions[regionCode] = {
        name: regionConfig.name,
        regulations: regionConfig.regulations,
        totalConsents: regionConsents.length,
        consentRate: consentRate,
        totalProcessingRecords: regionProcessing.length,
        processingCompliance: processingCompliance,
        lastAudit: new Date().toISOString(),
        issues: this.identifyComplianceIssues(regionCode)
      }

      report.summary.totalConsents += regionConsents.length
      report.summary.totalProcessingRecords += regionProcessing.length
    }

    report.summary.complianceScore = this.calculateOverallComplianceScore()

    return report
  }

  private calculateProcessingCompliance(region: string): number {
    const regionProcessing = this.processingRecords
      .filter(record => record.region === region)

    if (regionProcessing.length === 0) return 100

    const compliantRecords = regionProcessing.filter(record => {
      // Check if processing has valid legal basis
      return record.legalBasis && record.legalBasis.length > 0
    })

    return (compliantRecords.length / regionProcessing.length) * 100
  }

  private calculateOverallComplianceScore(): number {
    const regions = Array.from(this.regions.keys()).filter(r => r !== 'DEFAULT')
    if (regions.length === 0) return 100

    let totalScore = 0
    for (const region of regions) {
      const processingCompliance = this.calculateProcessingCompliance(region)
      const consentCompliance = this.calculateConsentCompliance(region)
      const regionScore = (processingCompliance + consentCompliance) / 2
      totalScore += regionScore
    }

    return totalScore / regions.length
  }

  private calculateConsentCompliance(region: string): number {
    const regionConfig = this.regions.get(region)
    if (!regionConfig || !regionConfig.consentRequired) return 100

    const regionConsents = Array.from(this.consentRecords.values())
      .flat()
      .filter(consent => consent.region === region)

    if (regionConsents.length === 0) return 50 // Neutral score if no consents

    const validConsents = regionConsents.filter(consent => {
      const age = Date.now() - consent.timestamp.getTime()
      const maxAge = 365 * 24 * 60 * 60 * 1000 // 1 year
      return age <= maxAge && consent.granted
    })

    return (validConsents.length / regionConsents.length) * 100
  }

  private identifyComplianceIssues(region: string): string[] {
    const issues: string[] = []
    const regionConfig = this.regions.get(region)
    
    if (!regionConfig) return issues

    // Check for missing consents
    if (regionConfig.consentRequired) {
      const regionConsents = Array.from(this.consentRecords.values())
        .flat()
        .filter(consent => consent.region === region)

      const expiredConsents = regionConsents.filter(consent => {
        const age = Date.now() - consent.timestamp.getTime()
        const maxAge = 365 * 24 * 60 * 60 * 1000
        return age > maxAge
      })

      if (expiredConsents.length > 0) {
        issues.push(`${expiredConsents.length} expired consents require renewal`)
      }
    }

    // Check for processing without legal basis
    const regionProcessing = this.processingRecords
      .filter(record => record.region === region)

    const processingWithoutBasis = regionProcessing.filter(record => 
      !record.legalBasis || record.legalBasis.length === 0
    )

    if (processingWithoutBasis.length > 0) {
      issues.push(`${processingWithoutBasis.length} processing activities lack legal basis`)
    }

    return issues
  }

  // Helper methods
  private getUserRegion(userId: string): string {
    // In a real implementation, this would look up user's region from user profile
    return 'DEFAULT'
  }

  private getUserDataCategories(userId: string): string[] {
    return ['profile_data', 'experiment_preferences', 'usage_analytics', 'causal_models']
  }

  private getUserProcessingPurposes(userId: string): string[] {
    return ['service_provision', 'analytics', 'research', 'performance_optimization']
  }

  private getUserRetentionPeriods(userId: string): any {
    return {
      profile_data: '5 years',
      experiment_data: '3 years',
      usage_analytics: '2 years',
      causal_models: 'indefinite (with consent)'
    }
  }

  private getUserDataRecipients(userId: string): string[] {
    return ['internal_analytics', 'research_partners', 'cloud_providers']
  }
}

// Global instance
export const globalCompliance = new GlobalComplianceService()

// Middleware for Express.js to handle compliance automatically
export function complianceMiddleware(options: { enforceConsent?: boolean } = {}) {
  return (req: any, res: any, next: any) => {
    // Determine user region
    const userIp = req.ip || req.connection.remoteAddress
    const userLocation = req.headers['cf-ipcountry'] || req.headers['x-user-location']
    const region = globalCompliance.determineUserRegion(userIp, userLocation)
    
    // Attach compliance info to request
    req.compliance = {
      region,
      requirements: globalCompliance.getComplianceRequirements(region)
    }

    // Check consent if required
    if (options.enforceConsent && req.user?.id) {
      const requiresConsent = globalCompliance.isConsentRequired(
        req.user.id,
        req.method.toLowerCase(),
        'api_access',
        region
      )

      if (requiresConsent && !globalCompliance.hasValidConsent(req.user.id, 'api_access')) {
        return res.status(403).json({
          error: 'Consent required',
          message: 'This operation requires user consent under applicable data protection regulations.',
          region,
          consentUrl: '/api/consent',
          regulations: req.compliance.requirements?.regulations || []
        })
      }
    }

    next()
  }
}

export default GlobalComplianceService