# Changelog - Pipeline V3

All notable changes to the Pipeline V3 redesign will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Created new branch `feature/pipeline-v3-redesign`
- Set up clean project structure in `pipeline_v3/` directory
- Created comprehensive documentation:
  - Main README with project overview and roadmap
  - FEATURE_FRAMEWORK.md with 150-180 feature specifications
  - DERIVED_XG.md with xG calculation methodology
- Added requirements.txt with all dependencies
- Added .env.example template for configuration
- Defined 3-pillar feature approach:
  - Pillar 1: Fundamentals (50 features)
  - Pillar 2: Modern Analytics (60 features)
  - Pillar 3: Hidden Edges (40 features)

### Design Decisions
- **Independence:** No external AI models or paid add-ons
- **Derived xG:** Calculate xG from base statistics (saves $1,800-3,600/year)
- **Clean Architecture:** Modular, testable, maintainable code
- **Target Performance:** 40-50% ROI, 68-72% win rate

### Next Steps
- [ ] Implement Elo rating calculator
- [ ] Build derived xG calculator
- [ ] Create SportMonks API client
- [ ] Set up database schema
- [ ] Begin feature engineering pipeline

---

## [0.1.0] - 2026-01-25

### Initial Setup
- Project structure created
- Documentation framework established
- Development roadmap defined
- Success metrics identified

---

**Version Naming:**
- 0.x.x = Development/Alpha
- 1.x.x = Production-ready
- 2.x.x = Major enhancements
