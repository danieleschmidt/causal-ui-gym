# Disaster Recovery Plan

Comprehensive disaster recovery procedures for Causal UI Gym.

## Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

| Component | RTO | RPO | Priority |
|-----------|-----|-----|----------|
| Frontend Application | 15 minutes | 1 hour | High |
| JAX Backend | 30 minutes | 15 minutes | Critical |
| Development Environment | 2 hours | 24 hours | Medium |
| CI/CD Pipeline | 1 hour | 4 hours | High |
| Documentation | 4 hours | 24 hours | Low |

## Backup Strategy

### Code Repository
- **Primary**: GitHub with automated backups
- **Secondary**: GitLab mirror (manual sync weekly)
- **Tertiary**: Local developer clones

### Dependencies and Packages
- **NPM packages**: package-lock.json ensures reproducible builds
- **Python packages**: requirements.txt with pinned versions
- **Container images**: Multi-registry storage (Docker Hub + GHCR)

### Configuration and Secrets
- **Infrastructure config**: Version controlled in this repository
- **Secrets**: Stored in GitHub Secrets with manual backup to secure location
- **Environment variables**: Documented in `.env.example`

## Disaster Scenarios and Response

### Scenario 1: GitHub Repository Unavailable

**Impact**: Development stops, CI/CD fails, documentation inaccessible

**Recovery Steps**:
1. **Immediate (0-15 minutes)**:
   - Switch to GitLab mirror: `git remote set-url origin https://gitlab.com/backup/causal-ui-gym.git`
   - Continue development on backup repository
   - Update team via Slack/Discord

2. **Short-term (15 minutes - 2 hours)**:
   - Set up temporary CI/CD on GitLab or local Jenkins
   - Update documentation links to point to backup
   - Communicate estimated recovery time to stakeholders

3. **Long-term (2+ hours)**:
   - Monitor GitHub status and restore when available
   - Sync changes back to primary repository
   - Update all references and links

### Scenario 2: Development Environment Corruption

**Impact**: Cannot build, test, or run the application

**Recovery Steps**:
1. **Quick Recovery (0-30 minutes)**:
   ```bash
   # Clean reset
   rm -rf node_modules/ .venv/ dist/ coverage/
   npm install
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   npm run build
   ```

2. **Container-based Recovery (5-15 minutes)**:
   ```bash
   # Use Docker for clean environment
   docker-compose down -v
   docker system prune -f
   docker-compose up --build
   ```

3. **Complete Environment Rebuild (30-60 minutes)**:
   - Fresh OS installation or VM reset
   - Follow `docs/DEVELOPMENT.md` setup guide
   - Restore from last known good configuration

### Scenario 3: CI/CD Pipeline Failure

**Impact**: Cannot deploy, automated testing stops, releases blocked

**Recovery Steps**:
1. **Immediate Assessment (0-10 minutes)**:
   - Check GitHub Actions status page
   - Review recent commits for breaking changes
   - Check if it's a service-wide issue

2. **Manual Deployment (10-30 minutes)**:
   ```bash
   # Manual build and test
   npm run build
   npm test
   python -m pytest tests/
   
   # Manual security scan
   python security-scan.py
   
   # Manual container build
   docker build -t causal-ui-gym .
   ```

3. **Alternative CI/CD (30-60 minutes)**:
   - Configure GitLab CI as backup
   - Set up local Jenkins instance
   - Use GitHub Actions alternatives (CircleCI, Travis)

### Scenario 4: Data Loss (Models, Experiments)

**Impact**: Loss of trained models, experimental results, user data

**Recovery Steps**:
1. **Immediate (0-15 minutes)**:
   - Stop all data-writing processes
   - Assess scope of data loss
   - Check if backups are available

2. **Recovery Process (15-60 minutes)**:
   ```bash
   # Check for local backups
   find . -name "*.ckpt" -o -name "*.pkl" -o -name "*.h5"
   
   # Restore from cloud storage if configured
   # aws s3 sync s3://causal-ui-gym-backups/ ./data/
   
   # Regenerate models if needed
   python scripts/retrain_models.py --fast-mode
   ```

3. **Prevention (Ongoing)**:
   - Implement automated model versioning
   - Set up cloud storage for critical data
   - Regular backup verification

## Recovery Procedures

### Environment Recovery Checklist

- [ ] Clone repository from backup source
- [ ] Install Node.js dependencies: `npm install`
- [ ] Set up Python environment: `python -m venv .venv && pip install -r requirements.txt`
- [ ] Verify environment: `npm run typecheck && python -m pytest tests/`
- [ ] Run security scan: `python security-scan.py`
- [ ] Test application: `npm run dev`
- [ ] Verify container build: `docker build -t causal-ui-gym .`

### Data Recovery Checklist

- [ ] Identify affected data types (models, configs, experiments)
- [ ] Check local backup locations
- [ ] Verify cloud backup integrity
- [ ] Restore data with verification
- [ ] Test restored data functionality
- [ ] Update backup procedures based on lessons learned

### Communication Plan

**Internal Team**:
1. Immediate notification via Slack/Discord
2. Status updates every 30 minutes during active recovery
3. Post-incident review within 24 hours

**External Stakeholders**:
1. Update status page if applicable
2. Email notification to key users
3. GitHub repository status update

## Testing and Validation

### Monthly Disaster Recovery Drills

1. **Week 1**: Test environment recovery from clean state
2. **Week 2**: Simulate CI/CD failure and manual deployment
3. **Week 3**: Test backup repository switch
4. **Week 4**: Full disaster scenario simulation

### Validation Criteria

After any recovery:
- All tests pass: `npm test && python -m pytest`
- Security scan passes: `python security-scan.py`
- Application builds and runs: `npm run build && npm run dev`
- Container builds successfully: `docker build -t causal-ui-gym .`
- Performance benchmarks within acceptable range

## Contact Information

**Primary Recovery Team**:
- Lead Developer: [GitHub username]
- DevOps Lead: [GitHub username]
- Security Contact: See SECURITY.md

**Escalation Path**:
1. Team Lead (0-30 minutes)
2. Technical Manager (30-60 minutes)
3. External Support (60+ minutes)

## Recovery Time Tracking

Document all recovery attempts:

```
Date: YYYY-MM-DD
Incident: [Brief description]
Detection Time: HH:MM
Recovery Start: HH:MM
Recovery Complete: HH:MM
Total Downtime: X hours Y minutes
Lessons Learned: [Key takeaways]
Action Items: [Prevention measures]
```

## Post-Incident Actions

1. **Immediate (0-24 hours)**:
   - Document incident timeline
   - Identify root cause
   - Implement immediate fixes

2. **Short-term (1-7 days)**:
   - Update disaster recovery procedures
   - Enhance monitoring and alerting
   - Conduct team retrospective

3. **Long-term (1-4 weeks)**:
   - Implement systemic improvements
   - Update backup strategies
   - Enhance automation

## Regular Maintenance

- **Weekly**: Verify backup integrity
- **Monthly**: Test recovery procedures
- **Quarterly**: Review and update this document
- **Annually**: Full disaster recovery exercise