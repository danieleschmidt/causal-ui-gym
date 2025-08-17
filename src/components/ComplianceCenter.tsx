/**
 * Compliance Center Component
 * 
 * Provides comprehensive regulatory compliance management including
 * GDPR, CCPA, SOX, HIPAA, accessibility standards, and audit trails.
 */

import React, { useState, useEffect, useMemo } from 'react'
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Chip,
  LinearProgress,
  Tabs,
  Tab,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  TableContainer,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Badge
} from '@mui/material'
import {
  Security as SecurityIcon,
  Gavel as GavelIcon,
  Shield as ShieldIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Check as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  AccessibilityNew as AccessibilityIcon,
  Language as LanguageIcon,
  Storage as StorageIcon,
  VpnLock as VpnLockIcon,
  Assignment as AssignmentIcon,
  Timeline as TimelineIcon,
  VerifiedUser as VerifiedUserIcon,
  Policy as PolicyIcon,
  Description as DescriptionIcon
} from '@mui/icons-material'
import { useI18n } from '../hooks/useI18n'

interface ComplianceStatus {
  standard: string
  status: 'compliant' | 'partial' | 'non-compliant' | 'pending'
  lastAudit: string
  nextAudit: string
  coverage: number
  requirements: ComplianceRequirement[]
  certifications: string[]
  documents: ComplianceDocument[]
}

interface ComplianceRequirement {
  id: string
  title: string
  description: string
  status: 'met' | 'partial' | 'not-met' | 'not-applicable'
  priority: 'high' | 'medium' | 'low'
  dueDate?: string
  evidence: string[]
  responsible: string
  lastUpdated: string
}

interface ComplianceDocument {
  id: string
  title: string
  type: 'policy' | 'procedure' | 'certificate' | 'report' | 'evidence'
  version: string
  status: 'draft' | 'approved' | 'expired'
  createdDate: string
  expiryDate?: string
  size: number
  downloadUrl: string
}

interface DataSubjectRequest {
  id: string
  type: 'access' | 'rectification' | 'erasure' | 'portability' | 'restriction' | 'objection'
  status: 'received' | 'in-progress' | 'completed' | 'rejected'
  submittedDate: string
  requestedBy: string
  email: string
  reason: string
  dueDate: string
  handledBy?: string
  response?: string
  completedDate?: string
}

interface AuditLog {
  id: string
  timestamp: string
  user: string
  action: string
  resource: string
  details: string
  ipAddress: string
  userAgent: string
  result: 'success' | 'failure' | 'warning'
  dataTypes: string[]
  lawfulBasis?: string
  retention?: string
}

interface PrivacySettings {
  dataMinimization: boolean
  purposeLimitation: boolean
  storageMinimization: boolean
  transparencyEnabled: boolean
  consentManagement: boolean
  cookieManagement: boolean
  dataProcessingLogging: boolean
  rightToErasure: boolean
  dataPortability: boolean
  privacyByDesign: boolean
  encryptionRequired: boolean
  anonymizationEnabled: boolean
}

interface AccessibilitySettings {
  wcagLevel: 'A' | 'AA' | 'AAA'
  screenReaderSupport: boolean
  keyboardNavigation: boolean
  highContrastMode: boolean
  reducedMotion: boolean
  audioDescriptions: boolean
  signLanguage: boolean
  captionsEnabled: boolean
  fontSize: 'small' | 'medium' | 'large' | 'extra-large'
  colorBlindnessSupport: boolean
  cognitiveSupport: boolean
  alternativeFormats: boolean
}

export default function ComplianceCenter() {
  const { t, locale, formatDate } = useI18n()
  
  const [activeTab, setActiveTab] = useState(0)
  const [complianceStatuses, setComplianceStatuses] = useState<ComplianceStatus[]>([])
  const [dataSubjectRequests, setDataSubjectRequests] = useState<DataSubjectRequest[]>([])
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([])
  const [privacySettings, setPrivacySettings] = useState<PrivacySettings>({
    dataMinimization: true,
    purposeLimitation: true,
    storageMinimization: true,
    transparencyEnabled: true,
    consentManagement: true,
    cookieManagement: true,
    dataProcessingLogging: true,
    rightToErasure: true,
    dataPortability: true,
    privacyByDesign: true,
    encryptionRequired: true,
    anonymizationEnabled: true
  })
  const [accessibilitySettings, setAccessibilitySettings] = useState<AccessibilitySettings>({
    wcagLevel: 'AA',
    screenReaderSupport: true,
    keyboardNavigation: true,
    highContrastMode: false,
    reducedMotion: false,
    audioDescriptions: false,
    signLanguage: false,
    captionsEnabled: true,
    fontSize: 'medium',
    colorBlindnessSupport: true,
    cognitiveSupport: true,
    alternativeFormats: true
  })

  const [, setSelectedCompliance] = useState<string | null>(null)
  const [requestDialog, setRequestDialog] = useState(false)
  const [newRequest, setNewRequest] = useState<Partial<DataSubjectRequest>>({})
  const [auditDialog, setAuditDialog] = useState(false)
  const [selectedAuditLog, setSelectedAuditLog] = useState<AuditLog | null>(null)

  // Initialize mock data
  useEffect(() => {
    setComplianceStatuses([
      {
        standard: 'GDPR',
        status: 'compliant',
        lastAudit: '2024-01-15',
        nextAudit: '2024-07-15',
        coverage: 95,
        requirements: [
          {
            id: 'gdpr-1',
            title: 'Data Processing Records',
            description: 'Maintain records of all data processing activities',
            status: 'met',
            priority: 'high',
            evidence: ['Processing register', 'Privacy policy'],
            responsible: 'Data Protection Officer',
            lastUpdated: '2024-01-10'
          },
          {
            id: 'gdpr-2',
            title: 'Consent Management',
            description: 'Implement proper consent collection and withdrawal mechanisms',
            status: 'met',
            priority: 'high',
            evidence: ['Consent management system', 'Cookie banner'],
            responsible: 'Privacy Team',
            lastUpdated: '2024-01-08'
          }
        ],
        certifications: ['ISO 27001', 'SOC 2 Type II'],
        documents: [
          {
            id: 'gdpr-policy',
            title: 'GDPR Compliance Policy',
            type: 'policy',
            version: '2.1',
            status: 'approved',
            createdDate: '2024-01-01',
            expiryDate: '2025-01-01',
            size: 2048000,
            downloadUrl: '/documents/gdpr-policy.pdf'
          }
        ]
      },
      {
        standard: 'CCPA',
        status: 'compliant',
        lastAudit: '2024-01-20',
        nextAudit: '2024-07-20',
        coverage: 88,
        requirements: [
          {
            id: 'ccpa-1',
            title: 'Consumer Rights Implementation',
            description: 'Implement all CCPA consumer rights',
            status: 'met',
            priority: 'high',
            evidence: ['Rights portal', 'Privacy policy updates'],
            responsible: 'Privacy Team',
            lastUpdated: '2024-01-15'
          }
        ],
        certifications: [],
        documents: []
      },
      {
        standard: 'WCAG 2.1 AA',
        status: 'partial',
        lastAudit: '2024-01-10',
        nextAudit: '2024-04-10',
        coverage: 82,
        requirements: [
          {
            id: 'wcag-1',
            title: 'Keyboard Navigation',
            description: 'All functionality must be keyboard accessible',
            status: 'met',
            priority: 'high',
            evidence: ['Accessibility audit report'],
            responsible: 'UI/UX Team',
            lastUpdated: '2024-01-05'
          },
          {
            id: 'wcag-2',
            title: 'Color Contrast',
            description: 'Ensure sufficient color contrast ratios',
            status: 'partial',
            priority: 'medium',
            evidence: ['Color audit report'],
            responsible: 'Design Team',
            lastUpdated: '2024-01-03'
          }
        ],
        certifications: [],
        documents: []
      }
    ])

    setDataSubjectRequests([
      {
        id: 'dsr-001',
        type: 'access',
        status: 'completed',
        submittedDate: '2024-01-20',
        requestedBy: 'John Smith',
        email: 'john.smith@example.com',
        reason: 'Want to review my personal data',
        dueDate: '2024-02-19',
        handledBy: 'Privacy Team',
        response: 'Data package provided via secure download',
        completedDate: '2024-01-25'
      },
      {
        id: 'dsr-002',
        type: 'erasure',
        status: 'in-progress',
        submittedDate: '2024-01-22',
        requestedBy: 'Sarah Johnson',
        email: 'sarah.johnson@example.com',
        reason: 'Account closure and data deletion',
        dueDate: '2024-02-21',
        handledBy: 'Privacy Team'
      }
    ])

    setAuditLogs([
      {
        id: 'audit-001',
        timestamp: '2024-01-22T14:30:00Z',
        user: 'admin@example.com',
        action: 'DATA_ACCESS',
        resource: 'User Profile',
        details: 'Accessed user profile data for ID: 12345',
        ipAddress: '192.168.1.100',
        userAgent: 'Mozilla/5.0...',
        result: 'success',
        dataTypes: ['Personal Data', 'Contact Information'],
        lawfulBasis: 'Consent',
        retention: '7 years'
      },
      {
        id: 'audit-002',
        timestamp: '2024-01-22T14:15:00Z',
        user: 'privacy@example.com',
        action: 'DATA_DELETION',
        resource: 'User Account',
        details: 'Deleted user account and associated data for DSR-002',
        ipAddress: '192.168.1.101',
        userAgent: 'Mozilla/5.0...',
        result: 'success',
        dataTypes: ['Personal Data', 'Usage Data'],
        lawfulBasis: 'Right to Erasure'
      }
    ])
  }, [])

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'compliant':
      case 'met':
      case 'completed':
        return <CheckIcon color="success" />
      case 'partial':
      case 'in-progress':
        return <WarningIcon color="warning" />
      case 'non-compliant':
      case 'not-met':
      case 'rejected':
        return <ErrorIcon color="error" />
      default:
        return <InfoIcon color="info" />
    }
  }

  const getStatusColor = (status: string): 'success' | 'warning' | 'error' | 'info' => {
    switch (status) {
      case 'compliant':
      case 'met':
      case 'completed':
        return 'success'
      case 'partial':
      case 'in-progress':
        return 'warning'
      case 'non-compliant':
      case 'not-met':
      case 'rejected':
        return 'error'
      default:
        return 'info'
    }
  }

  const handleCreateDataSubjectRequest = () => {
    if (newRequest.type && newRequest.requestedBy && newRequest.email) {
      const request: DataSubjectRequest = {
        id: `dsr-${Date.now()}`,
        type: newRequest.type as DataSubjectRequest['type'],
        status: 'received',
        submittedDate: new Date().toISOString(),
        requestedBy: newRequest.requestedBy,
        email: newRequest.email,
        reason: newRequest.reason || '',
        dueDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString()
      }
      
      setDataSubjectRequests(prev => [request, ...prev])
      setRequestDialog(false)
      setNewRequest({})
    }
  }

  const exportComplianceReport = () => {
    const report = {
      generatedAt: new Date().toISOString(),
      complianceStatuses,
      dataSubjectRequests,
      privacySettings,
      accessibilitySettings,
      auditLogsSummary: {
        totalEntries: auditLogs.length,
        timeRange: {
          from: auditLogs[auditLogs.length - 1]?.timestamp,
          to: auditLogs[0]?.timestamp
        }
      }
    }
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `compliance-report-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const overallComplianceScore = useMemo(() => {
    if (complianceStatuses.length === 0) return 0
    return Math.round(
      complianceStatuses.reduce((sum, status) => sum + status.coverage, 0) / complianceStatuses.length
    )
  }, [complianceStatuses])

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SecurityIcon color="primary" />
          Compliance & Privacy Center
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Comprehensive regulatory compliance and privacy management
        </Typography>
      </Box>

      {/* Overall Compliance Score */}
      <Paper sx={{ p: 3, mb: 3, bgcolor: 'primary.50' }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={8}>
            <Typography variant="h6" gutterBottom>
              Overall Compliance Score
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
              <LinearProgress 
                variant="determinate" 
                value={overallComplianceScore} 
                sx={{ flexGrow: 1, height: 10, borderRadius: 5 }}
              />
              <Typography variant="h6" color="primary">
                {overallComplianceScore}%
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Based on {complianceStatuses.length} compliance standards
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={exportComplianceReport}
              >
                Export Report
              </Button>
              <Button
                variant="contained"
                startIcon={<AssignmentIcon />}
                onClick={() => setRequestDialog(true)}
              >
                New DSR
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Quick Stats */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Active Standards
                  </Typography>
                  <Typography variant="h4">
                    {complianceStatuses.length}
                  </Typography>
                </Box>
                <GavelIcon color="primary" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Pending DSRs
                  </Typography>
                  <Typography variant="h4">
                    {dataSubjectRequests.filter(r => r.status === 'received' || r.status === 'in-progress').length}
                  </Typography>
                </Box>
                <AssignmentIcon color="warning" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Audit Events (24h)
                  </Typography>
                  <Typography variant="h4">
                    {auditLogs.filter(log => 
                      new Date(log.timestamp) > new Date(Date.now() - 24 * 60 * 60 * 1000)
                    ).length}
                  </Typography>
                </Box>
                <TimelineIcon color="info" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Certifications
                  </Typography>
                  <Typography variant="h4">
                    {complianceStatuses.reduce((sum, status) => sum + status.certifications.length, 0)}
                  </Typography>
                </Box>
                <VerifiedUserIcon color="success" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} variant="fullWidth">
          <Tab label="Compliance Standards" icon={<GavelIcon />} />
          <Tab label="Data Subject Requests" icon={<AssignmentIcon />} />
          <Tab label="Privacy Settings" icon={<ShieldIcon />} />
          <Tab label="Accessibility" icon={<AccessibilityIcon />} />
          <Tab label="Audit Logs" icon={<TimelineIcon />} />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {activeTab === 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Compliance Standards Status
          </Typography>
          <Grid container spacing={2}>
            {complianceStatuses.map((compliance) => (
              <Grid item xs={12} md={6} lg={4} key={compliance.standard}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h6">{compliance.standard}</Typography>
                      <Chip
                        icon={getStatusIcon(compliance.status)}
                        label={compliance.status}
                        color={getStatusColor(compliance.status)}
                        size="small"
                      />
                    </Box>
                    
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Coverage: {compliance.coverage}%
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={compliance.coverage} 
                        sx={{ mb: 1 }}
                      />
                    </Box>

                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Last Audit: {formatDate(new Date(compliance.lastAudit))}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Next Audit: {formatDate(new Date(compliance.nextAudit))}
                    </Typography>
                    
                    {compliance.certifications.length > 0 && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="body2" gutterBottom>Certifications:</Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {compliance.certifications.map((cert, index) => (
                            <Chip key={index} label={cert} size="small" variant="outlined" />
                          ))}
                        </Box>
                      </Box>
                    )}
                  </CardContent>
                  
                  <CardActions>
                    <Button size="small" onClick={() => setSelectedCompliance(compliance.standard)}>
                      View Details
                    </Button>
                    <Button size="small" startIcon={<DownloadIcon />}>
                      Export
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {activeTab === 1 && (
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Data Subject Requests</Typography>
            <Button
              variant="contained"
              startIcon={<AssignmentIcon />}
              onClick={() => setRequestDialog(true)}
            >
              New Request
            </Button>
          </Box>

          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Request ID</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Requester</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Submitted</TableCell>
                  <TableCell>Due Date</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {dataSubjectRequests.map((request) => (
                  <TableRow key={request.id}>
                    <TableCell>{request.id}</TableCell>
                    <TableCell>
                      <Chip label={request.type} size="small" variant="outlined" />
                    </TableCell>
                    <TableCell>{request.requestedBy}</TableCell>
                    <TableCell>
                      <Chip
                        icon={getStatusIcon(request.status)}
                        label={request.status}
                        color={getStatusColor(request.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{formatDate(new Date(request.submittedDate))}</TableCell>
                    <TableCell>{formatDate(new Date(request.dueDate))}</TableCell>
                    <TableCell>
                      <IconButton size="small">
                        <VisibilityIcon />
                      </IconButton>
                      <IconButton size="small">
                        <EditIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      {activeTab === 2 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Privacy Settings
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Data Processing Controls
                </Typography>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <StorageIcon />
                    </ListItemIcon>
                    <ListItemText primary="Data Minimization" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={privacySettings.dataMinimization}
                          onChange={(e) => setPrivacySettings(prev => ({
                            ...prev,
                            dataMinimization: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <VpnLockIcon />
                    </ListItemIcon>
                    <ListItemText primary="Purpose Limitation" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={privacySettings.purposeLimitation}
                          onChange={(e) => setPrivacySettings(prev => ({
                            ...prev,
                            purposeLimitation: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <SecurityIcon />
                    </ListItemIcon>
                    <ListItemText primary="Encryption Required" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={privacySettings.encryptionRequired}
                          onChange={(e) => setPrivacySettings(prev => ({
                            ...prev,
                            encryptionRequired: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                </List>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  User Rights Management
                </Typography>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <DeleteIcon />
                    </ListItemIcon>
                    <ListItemText primary="Right to Erasure" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={privacySettings.rightToErasure}
                          onChange={(e) => setPrivacySettings(prev => ({
                            ...prev,
                            rightToErasure: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <DownloadIcon />
                    </ListItemIcon>
                    <ListItemText primary="Data Portability" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={privacySettings.dataPortability}
                          onChange={(e) => setPrivacySettings(prev => ({
                            ...prev,
                            dataPortability: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <VisibilityIcon />
                    </ListItemIcon>
                    <ListItemText primary="Transparency Enabled" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={privacySettings.transparencyEnabled}
                          onChange={(e) => setPrivacySettings(prev => ({
                            ...prev,
                            transparencyEnabled: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                </List>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      )}

      {activeTab === 3 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Accessibility Settings
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  WCAG Compliance
                </Typography>
                
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>WCAG Level</InputLabel>
                  <Select
                    value={accessibilitySettings.wcagLevel}
                    onChange={(e) => setAccessibilitySettings(prev => ({
                      ...prev,
                      wcagLevel: e.target.value as AccessibilitySettings['wcagLevel']
                    }))}
                  >
                    <MenuItem value="A">WCAG 2.1 Level A</MenuItem>
                    <MenuItem value="AA">WCAG 2.1 Level AA</MenuItem>
                    <MenuItem value="AAA">WCAG 2.1 Level AAA</MenuItem>
                  </Select>
                </FormControl>
                
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <AccessibilityIcon />
                    </ListItemIcon>
                    <ListItemText primary="Screen Reader Support" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={accessibilitySettings.screenReaderSupport}
                          onChange={(e) => setAccessibilitySettings(prev => ({
                            ...prev,
                            screenReaderSupport: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <LanguageIcon />
                    </ListItemIcon>
                    <ListItemText primary="Keyboard Navigation" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={accessibilitySettings.keyboardNavigation}
                          onChange={(e) => setAccessibilitySettings(prev => ({
                            ...prev,
                            keyboardNavigation: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                </List>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Visual & Cognitive Support
                </Typography>
                
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Font Size</InputLabel>
                  <Select
                    value={accessibilitySettings.fontSize}
                    onChange={(e) => setAccessibilitySettings(prev => ({
                      ...prev,
                      fontSize: e.target.value as AccessibilitySettings['fontSize']
                    }))}
                  >
                    <MenuItem value="small">Small</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="large">Large</MenuItem>
                    <MenuItem value="extra-large">Extra Large</MenuItem>
                  </Select>
                </FormControl>
                
                <List>
                  <ListItem>
                    <ListItemText primary="High Contrast Mode" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={accessibilitySettings.highContrastMode}
                          onChange={(e) => setAccessibilitySettings(prev => ({
                            ...prev,
                            highContrastMode: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText primary="Reduced Motion" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={accessibilitySettings.reducedMotion}
                          onChange={(e) => setAccessibilitySettings(prev => ({
                            ...prev,
                            reducedMotion: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText primary="Color Blindness Support" />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={accessibilitySettings.colorBlindnessSupport}
                          onChange={(e) => setAccessibilitySettings(prev => ({
                            ...prev,
                            colorBlindnessSupport: e.target.checked
                          }))}
                        />
                      }
                      label=""
                    />
                  </ListItem>
                </List>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      )}

      {activeTab === 4 && (
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Audit Logs</Typography>
            <Button
              variant="outlined"
              startIcon={<DownloadIcon />}
              onClick={() => {
                const blob = new Blob([JSON.stringify(auditLogs, null, 2)], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `audit-logs-${new Date().toISOString().split('T')[0]}.json`
                document.body.appendChild(a)
                a.click()
                document.body.removeChild(a)
                URL.revokeObjectURL(url)
              }}
            >
              Export Logs
            </Button>
          </Box>

          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>User</TableCell>
                  <TableCell>Action</TableCell>
                  <TableCell>Resource</TableCell>
                  <TableCell>Result</TableCell>
                  <TableCell>Data Types</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {auditLogs.map((log) => (
                  <TableRow key={log.id}>
                    <TableCell>{formatDate(new Date(log.timestamp))}</TableCell>
                    <TableCell>{log.user}</TableCell>
                    <TableCell>{log.action}</TableCell>
                    <TableCell>{log.resource}</TableCell>
                    <TableCell>
                      <Chip
                        icon={getStatusIcon(log.result)}
                        label={log.result}
                        color={getStatusColor(log.result)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {log.dataTypes.map((type, index) => (
                          <Chip key={index} label={type} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <IconButton 
                        size="small"
                        onClick={() => {
                          setSelectedAuditLog(log)
                          setAuditDialog(true)
                        }}
                      >
                        <VisibilityIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      {/* Data Subject Request Dialog */}
      <Dialog open={requestDialog} onClose={() => setRequestDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create Data Subject Request</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Request Type</InputLabel>
                <Select
                  value={newRequest.type || ''}
                  onChange={(e) => setNewRequest(prev => ({ ...prev, type: e.target.value as DataSubjectRequest['type'] }))}
                >
                  <MenuItem value="access">Data Access</MenuItem>
                  <MenuItem value="rectification">Data Rectification</MenuItem>
                  <MenuItem value="erasure">Data Erasure</MenuItem>
                  <MenuItem value="portability">Data Portability</MenuItem>
                  <MenuItem value="restriction">Processing Restriction</MenuItem>
                  <MenuItem value="objection">Processing Objection</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Requester Name"
                value={newRequest.requestedBy || ''}
                onChange={(e) => setNewRequest(prev => ({ ...prev, requestedBy: e.target.value }))}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Email Address"
                type="email"
                value={newRequest.email || ''}
                onChange={(e) => setNewRequest(prev => ({ ...prev, email: e.target.value }))}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Reason/Details"
                value={newRequest.reason || ''}
                onChange={(e) => setNewRequest(prev => ({ ...prev, reason: e.target.value }))}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRequestDialog(false)}>Cancel</Button>
          <Button onClick={handleCreateDataSubjectRequest} variant="contained">
            Create Request
          </Button>
        </DialogActions>
      </Dialog>

      {/* Audit Log Details Dialog */}
      <Dialog open={auditDialog} onClose={() => setAuditDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Audit Log Details</DialogTitle>
        <DialogContent>
          {selectedAuditLog && (
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">Timestamp</Typography>
                <Typography variant="body2">{formatDate(new Date(selectedAuditLog.timestamp))}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">User</Typography>
                <Typography variant="body2">{selectedAuditLog.user}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">Action</Typography>
                <Typography variant="body2">{selectedAuditLog.action}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">Resource</Typography>
                <Typography variant="body2">{selectedAuditLog.resource}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2">Details</Typography>
                <Typography variant="body2">{selectedAuditLog.details}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">IP Address</Typography>
                <Typography variant="body2">{selectedAuditLog.ipAddress}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2">Result</Typography>
                <Chip
                  icon={getStatusIcon(selectedAuditLog.result)}
                  label={selectedAuditLog.result}
                  color={getStatusColor(selectedAuditLog.result)}
                  size="small"
                />
              </Grid>
              {selectedAuditLog.lawfulBasis && (
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Lawful Basis</Typography>
                  <Typography variant="body2">{selectedAuditLog.lawfulBasis}</Typography>
                </Grid>
              )}
              {selectedAuditLog.retention && (
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Retention Period</Typography>
                  <Typography variant="body2">{selectedAuditLog.retention}</Typography>
                </Grid>
              )}
              <Grid item xs={12}>
                <Typography variant="subtitle2">Data Types</Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                  {selectedAuditLog.dataTypes.map((type, index) => (
                    <Chip key={index} label={type} size="small" variant="outlined" />
                  ))}
                </Box>
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAuditDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}