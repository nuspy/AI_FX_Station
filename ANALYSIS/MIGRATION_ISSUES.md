# UI Migration Issues & Questions

**Project:** ForexGPT UI Migration  
**Start Date:** 2025-10-08  
**Status:** In Progress

Use this file to track issues, questions, and decisions during the migration process.

---

## Open Issues

### Issue #1 - [TITLE]
**Date:** YYYY-MM-DD  
**Priority:** High/Medium/Low  
**Component:** [service/mixin name]  
**Description:**
[Describe the issue]

**Proposed Solution:**
[Your proposed fix]

**Status:** Open/In Progress/Resolved  
**Resolved Date:** [if resolved]

---

## Questions

### Q1: [QUESTION]
**Asked:** YYYY-MM-DD  
**Asker:** [Name]  
**Context:** [Background]

**Answer:** [If answered]  
**Answered By:** [Name]  
**Date:** YYYY-MM-DD

---

## Architecture Decisions

### AD1: Service Responsibility Boundaries
**Date:** 2025-10-08  
**Decision:** 
- DataService handles all data loading/caching
- PlotService handles all visualization
- InteractionService handles mouse/keyboard input
- TradingService handles orders/positions/trades

**Rationale:**
Clear separation of concerns following single responsibility principle.

---

### AD2: Matplotlib Compatibility Layer
**Date:** 2025-10-08  
**Decision:**
Keep matplotlib as fallback for legacy indicators while primary rendering uses finplot/PyQtGraph.

**Rationale:**
Some indicator libraries may depend on matplotlib. Dual support ensures compatibility.

---

## Migration Blockers

### Blocker #1 - [TITLE]
**Blocking:** [What feature/phase is blocked]  
**Description:** [What's blocking progress]  
**Mitigation:** [How to work around]  
**Status:** Blocking/Resolved

---

## Code Review Notes

### Review #1 - [Component Name]
**Reviewer:** [Name]  
**Date:** YYYY-MM-DD  
**Files Reviewed:**
- file1.py
- file2.py

**Findings:**
- [ ] Issue 1
- [ ] Issue 2

**Action Items:**
- [ ] Fix X
- [ ] Refactor Y

---

## Testing Issues

### Test Failure #1 - [Test Name]
**Date:** YYYY-MM-DD  
**Test File:** tests/path/to/test.py  
**Error Message:**
```
[paste error]
```

**Root Cause:** [If known]  
**Fix:** [How was it fixed]  
**Status:** Open/Fixed

---

## Performance Issues

### Perf #1 - [Issue Title]
**Date:** YYYY-MM-DD  
**Symptom:** [What's slow]  
**Metrics:** [Before/after numbers]  
**Solution:** [How fixed]  
**Status:** Open/Resolved

---

## Breaking Changes

### BC1: [Change Description]
**Date:** YYYY-MM-DD  
**Affects:** [What code is affected]  
**Migration Path:** [How to update]  
**Documented:** Yes/No

---

## Technical Debt

### TD1: [Debt Item]
**Created:** YYYY-MM-DD  
**Description:** [What needs improvement]  
**Impact:** High/Medium/Low  
**Effort:** High/Medium/Low  
**Priority:** [For addressing]

---

## Lessons Learned

### Lesson #1 - [Topic]
**Date:** YYYY-MM-DD  
**Context:** [What happened]  
**Learning:** [What we learned]  
**Application:** [How to apply in future]

---

## Useful Resources

- **Full Migration Doc:** `UI_Migration.md`
- **Quick Reference:** `QUICK_REFERENCE.md`
- **Code Standards:** `../CONTRIBUTING.md`
- **Testing Guide:** `../tests/README.md`

---

## Contact Information

**Migration Lead:** [Name]  
**Technical Lead:** [Name]  
**Code Reviews:** [Team/Person]  
**Questions:** [How to reach out]

---

**Last Updated:** 2025-10-08
