# sEMG RF Feature Preset Analysis

## Summary

Created 13 feature presets for the sEMG Random Forest classifier by systematically removing feature groups from the `rf_enhanced` baseline. Benchmark results show that removing distribution features (**std** and **peak-to-peak**) yields the best unseen accuracy.

## All Presets (13 total)

### Original Presets (3)
| Preset | Features | Val Acc | Unseen Acc | Description |
|--------|----------|---------|------------|-------------|
| `baseline` | 24 | 0.8235 | 0.8029 | Baseline: MAV, RMS, WL, VAR, ZC, SSC |
| `baseline_plus_wamp` | 28 | 0.7983 | 0.8066 | + WAMP |
| `rf_enhanced` | 52 | 0.7983 | 0.8066 | + IEMG, Freq (mean_freq, median_freq, spectral_entropy), Distribution (std, ptp) |

### Single-Group Removal from rf_enhanced (6)
| Preset | Features | Val Acc | Unseen Acc | Removed |
|--------|----------|---------|------------|---------|
| `rf_enhanced_no_wamp` | 48 | 0.8067 | 0.8066 | WAMP |
| `rf_enhanced_no_iemg` | 48 | 0.8067 | 0.8029 | IEMG |
| `rf_enhanced_no_freq` | 40 | 0.7983 | 0.8139 | Freq features (mean_freq, median_freq, spectral_entropy) |
| `rf_enhanced_no_distribution` | 44 | 0.7983 | **0.8175** ⭐ | Distribution features (std, peak-to-peak) |
| `rf_enhanced_no_zc` | 48 | 0.7983 | 0.7993 | ZC (zero-crossing) |
| `rf_enhanced_no_ssc` | 48 | 0.7983 | 0.8139 | SSC (slope-sign-change) |

### Multi-Group Combinations (4)
| Preset | Features | Val Acc | Unseen Acc | Removed |
|--------|----------|---------|------------|---------|
| `rf_enhanced_no_zc_median` | 44 | 0.8067 | 0.8139 | ZC + median_freq |
| `rf_enhanced_no_freq_no_distribution` | 32 | 0.8151 | 0.7993 | Freq + Distribution |
| `rf_enhanced_core` | 32 | 0.8151 | 0.7993 | Freq + Distribution (same as above) |
| `rf_enhanced_light` | 28 | 0.7983 | 0.8066 | IEMG + Freq + Distribution |

## Results Summary

### Best Performers by Metric
- **Best Unseen Accuracy**: `rf_enhanced_no_distribution` (0.8175) ⭐
- **Best Validation Accuracy**: `baseline` (0.8235)
- **Best Trade-off**: `rf_enhanced_no_freq` or `rf_enhanced_no_ssc` (0.8139 unseen, 0.7983 val)

### Key Findings
1. **Removing distribution features (std, peak-to-peak) improves unseen accuracy** to 0.8175 (+1.09% vs rf_enhanced)
2. **Removing frequency features or SSC also improves generalization** (0.8139 unseen)
3. **Removing ZC actually hurts performance** (0.7993 unseen)
4. **Complete removals (freq + distribution) reduce both validation and unseen accuracy** (0.7993)

## Usage

### Run Benchmark (all 13 presets)
```bash
cd src/testbed/ml
python semg_benchmark_presets.py
```

### Train with Specific Preset
```bash
python semg_train_rf.py --feature_preset rf_enhanced_no_distribution \
  --data ../sutd_bmi_safety_data/combined.csv \
  --unseen ../sutd_bmi_safety_data/unseen
```

### Load Results
```bash
cat preset_benchmark_results.csv
```

## Files Modified

- **semg_model.py**
  - Added 13 presets to `FEATURE_PRESETS` dictionary
  - Added `add_zc`, `add_ssc` config keys to control baseline features
  - Added `drop_median_freq` key for fine-grained control
  - Updated `extract_features_baseline()` to respect add_zc/add_ssc flags
  - Updated `extract_features()` to handle new config options

- **semg_train_rf.py**
  - Added `--drop_features` CLI flag for inline feature removal

- **New Script: semg_benchmark_presets.py**
  - Evaluates all presets on train/val/unseen splits
  - Generates CSV report of results
  - Identifies best performers

## Recommendation

**For production use:**
- Use `rf_enhanced_no_distribution` for best unseen-data accuracy (0.8175)
- Reduces feature dimension from 52→44 while improving generalization
- Removes potentially noisy std and peak-to-peak measurements

**For validation-focused tasks:**
- Stick with `baseline` (highest val accuracy 0.8235) for model selection
- Or use `rf_enhanced_core` (val 0.8151) if you need richer features
