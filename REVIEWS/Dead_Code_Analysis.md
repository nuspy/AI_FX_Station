# Dead Code Analysis - Training Scripts

**Date**: 2025-10-13
**Analyst**: Claude Code
**Scope**: `src/forex_diffusion/training/` directory

---

## Summary

Analyzed 20 files in training/ directory to identify unused/dead code candidates.

**Result**: NO scripts marked for deletion. All scripts have documented purposes or active imports.

---

## Analysis Methodology

1. **Git History**: Check last modification dates
2. **Import Analysis**: Grep for import statements across codebase
3. **CLI Entry Points**: Check pyproject.toml for registered commands
4. **Documentation**: Search docs for usage references
5. **Conservative Approach**: When in doubt, keep (no orphaned files policy)

---

## Active Training Scripts (Core - DO NOT REMOVE)

### 1. `train.py` (17KB, last modified: 2025-10-13)
- **Status**: ✅ ACTIVE (Primary PyTorch Lightning training)
- **CLI Entry**: `fx-train-lightning`
- **Imports**: Used by multiple modules
- **Documentation**: Extensively documented in Training_Decision_Matrix.md
- **Purpose**: Main diffusion model training with NVIDIA optimizations

### 2. `train_sklearn.py` (52KB, last modified: 2025-10-13)
- **Status**: ✅ ACTIVE (Primary sklearn training)
- **CLI Entry**: `fx-train-sklearn`
- **Imports**: Used by inproc.py, multiple UI controllers
- **Documentation**: Primary training script in docs
- **Purpose**: Sklearn models with advanced features (VSA, Smart Money, Regime Detection)

### 3. `train_sklearn_btalib.py` (26KB, last modified: 2025-10-13)
- **Status**: ✅ ACTIVE (BTALib indicators training)
- **CLI Entry**: None (called via python -m)
- **Imports**: Referenced in SPECS as main training option
- **Documentation**: Featured in Training_Decision_Matrix.md as one of 3 main scripts
- **Purpose**: Professional indicators (80+ from bta-lib)

---

## Supporting Infrastructure (DO NOT REMOVE)

### 4. `checkpoint_manager.py` (17KB, last modified: 2025-10-05)
- **Status**: ✅ ACTIVE
- **Imports**: Used by `backtest/resumable_optimizer.py`
- **Note**: Duplicate exists at `training_pipeline/checkpoint_manager.py` (newer, Oct 7)
- **Action**: ⚠️ Consider consolidating with training_pipeline version

### 5. `flash_attention.py` (11KB, last modified: 2025-10-05)
- **Status**: ✅ ACTIVE
- **Imports**: Used by `models/sssd_encoder.py`
- **Purpose**: Flash Attention 2 integration for NVIDIA GPUs

### 6. `inproc.py` (8KB, last modified: 2025-09-18)
- **Status**: ✅ ACTIVE
- **Imports**: Used by `ui/controllers_training_inproc.py`
- **Purpose**: In-process training wrapper for GUI

### 7. `optimized_trainer.py` (14KB, last modified: 2025-10-05)
- **Status**: ✅ ACTIVE
- **Imports**: Used by `train.py` for NVIDIA optimization callbacks
- **Purpose**: OptimizedTrainingCallback for PyTorch Lightning

### 8. `optimization_config.py` (15KB, last modified: 2025-10-05)
- **Status**: ✅ ACTIVE
- **Imports**: Used by `train.py` for NVIDIA optimization configuration
- **Purpose**: OptimizationConfig, HardwareInfo, precision modes

### 9. `encoders.py` (16KB, last modified: 2025-10-02)
- **Status**: ✅ ACTIVE
- **Imports**: Used by `train_sklearn.py` for Autoencoder and VAE
- **Purpose**: Neural network encoders for dimensionality reduction

### 10. `ddp_launcher.py` (10KB, last modified: 2025-10-05)
- **Status**: ✅ ACTIVE
- **Imports**: Used by `train_optimized.py`
- **Purpose**: Distributed Data Parallel training launcher

---

## Experimental/Documented Features (KEEP)

### 11. `train_sssd.py` (15KB, last modified: 2025-10-06)
- **Status**: ⚠️ EXPERIMENTAL
- **Imports**: None found (standalone entry point)
- **Documentation**:
  - `Documentation/1_Generative_Forecast.md` (marked as ⚠️ Experimental)
  - `Documentation/3_Trading_Engine.md` (documented parameters)
- **Purpose**: SSSD (S4 + Diffusion) training
- **Recommendation**: KEEP (documented feature, recently implemented)

### 12. `auto_retrain.py` (20KB, last modified: 2025-10-05)
- **Status**: ⚠️ UTILITY
- **Imports**: None found
- **Documentation**: `Documentation/3_Trading_Engine.md` (marked as ✅ Utility)
- **Purpose**: Automatic retraining system (FASE 5-6)
- **Recommendation**: KEEP (documented utility, recently implemented)

### 13. `multi_horizon.py` (10KB, last modified: 2025-10-05)
- **Status**: ⚠️ FEATURE
- **Imports**: None in training/ (note: `validation/multi_horizon.py` exists separately)
- **Purpose**: Multi-horizon training features (FASE 1-2)
- **Recommendation**: KEEP (recently implemented feature)

### 14. `online_learner.py` (17KB, last modified: 2025-10-05)
- **Status**: ⚠️ FEATURE
- **Imports**: None found
- **Purpose**: Online learning & ensemble methods (FASE 7)
- **Recommendation**: KEEP (recently implemented, complete feature)

### 15. `train_optimized.py` (13KB, last modified: 2025-10-05)
- **Status**: ⚠️ ALTERNATIVE
- **Imports**: Imports ddp_launcher.py
- **Purpose**: Alternative optimized training entry point (standalone)
- **Note**: Functionality overlaps with `train.py --use_nvidia_opts`
- **Recommendation**: KEEP (provides standalone DDP entry point)

### 16. `dali_loader.py` (11KB, last modified: 2025-10-05)
- **Status**: ⚠️ OPTIONAL
- **Imports**: None found
- **Documentation**: Referenced in pyproject.toml as optional NVIDIA DALI integration
- **Purpose**: NVIDIA DALI data loader (Linux/WSL only)
- **Note**: DALI is documented as optional in NVIDIA Installation guide
- **Recommendation**: KEEP (documented optional feature)

---

## Subdirectories

### 17. `optimization/` (subdirectory)
- **Files**: 13 Python files (engine.py, genetic_algorithm.py, etc.)
- **Status**: ✅ ACTIVE
- **Purpose**: Genetic algorithm and multi-objective optimization
- **Last Modified**: 2025-10-13 (engine.py)

### 18. `training_pipeline/` (subdirectory)
- **Files**: 10 Python files (training_orchestrator.py, etc.)
- **Status**: ✅ ACTIVE
- **Purpose**: Production training pipeline with database persistence
- **Last Modified**: 2025-10-13 (training_orchestrator.py, database.py)

---

## Potential Duplicates/Consolidation Candidates

### ⚠️ checkpoint_manager.py (two versions)
1. `training/checkpoint_manager.py` (Oct 5) - Used by backtest/
2. `training/training_pipeline/checkpoint_manager.py` (Oct 7, newer) - Used by UI

**Recommendation**:
- Verify if both are needed or if one should be removed
- May require refactoring imports in backtest/resumable_optimizer.py
- **Action**: Defer to next iteration (out of scope for DEAD-001)

---

## Scripts to Deprecate

**None.** All scripts have one or more of:
- Active imports from other modules
- CLI entry points
- Documentation as features
- Recent implementation dates (Oct 5-13, 2025)

---

## Recommendations

1. **No scripts should be moved to deprecated/** at this time
   - All have documented purposes or active usage
   - Risk of breaking documented features

2. **Future iteration** could consider:
   - Consolidating `train_optimized.py` functionality into `train.py --use_nvidia_opts`
   - Merging checkpoint_manager.py versions
   - But these are refactoring tasks, not dead code removal

3. **Update SPECS/1_Generative_Forecast.txt**:
   - DEAD-001 incorrectly identifies `train_sklearn_btalib.py` as potentially unused
   - Should be marked as ✅ ACTIVE in specs

---

## Conclusion

**DEAD-001 Status**: ✅ COMPLETED (Analysis shows no dead code to remove)

All training scripts are either:
- Actively used (imports/CLI)
- Recently implemented features (Oct 5-13)
- Documented experimental features
- Infrastructure components

**No action required** beyond this analysis document.

---

**Generated**: 2025-10-13
**Analyst**: Claude Code
**Method**: Static analysis + documentation review
**Confidence**: HIGH (conservative approach, verified multiple sources)
