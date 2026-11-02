# DROID + ABC midtrain: 0.6s windows

User authorized both sources at0.6s anchor spacing, Video/Action peakLR1e-4,500 optimizer-update warmup, refreshed data/norm, one short16-H100 smoke and then formal one epoch. Target runtime approximately3days, not silent truncation of the epoch. Prior job32169 was cancelled before formal updates; old artifacts remain untouched.

Source baseline680c568 on personal frank/eighteen-preparation. Existing unrelated local modifications remain unstaged. H100 Frank identity verified; no jobs running at preparation start; shared GPFS20T available (not reserved personal quota).

## Necessary changes

- DROID enumerates each originally represented episode once, reuses its full10Hz sampling/Lance rows, and rebuilds0.6s starts from its original first accepted anchor. Rechecks sampling SHA, row/capture/source monotonicity, original per-camera and cross-camera tolerances. Rejected windows are explicitly receipted; original16 trajectory exclusions remain. Actual state target bounds remain checked before final statistics/admission.
- ABC retains original train/val stereo episode membership and physical17-frame timeline. A small adapter maps each episode's window ordinal to anchor ordinal*6. Source phases still rotate by NEW global window ordinal, retaining all three phases rather than aliasing every6th old index to one phase.
- Orion's norm function hardcodes dense anchors. Preparation reuses its pinned783b602d implementation with only anchor-count/selection/phase-index changes. Original RunningMoments, interpolation, masks and relative transforms remain unchanged. Stats carry stride metadata; readers reject old-stride stats.
- Existing scheduler only supported ratio warmup and a world-size-dependent tiny startLR. Added optional explicit warmup_steps/start_factor; this recipe uses500/.1. Legacy configurations retain prior behavior. CPU test simulates16 scheduler ticks per optimizer update and checks1e-5 start,1e-4 after500updates,1e-6 at horizon.

## Parameter audit

Keep perGPUbatch2,accum4,global128,ZeRO2,BF16,gradient checkpointing,Video/Action weights1/1,clip1,AdamWbetas(.9,.95)/weight_decay.01. The beta2 differs from Orion(.999) but is a valid existing OpenWAM default, not an identified bug. Frozen Stage3u8000/T5 and random ActionDiT remain. One-epoch scheduler horizon is computed from final admitted sizes; validation500optimizersteps,128/source; full saves10000microsteps=2500optimizerupdates plus final. All full states retained, no automatic deletion. Existing source-level mask and final-state paths were inspected; runtime verification still pending.

Seven targeted CPU contract tests passed: fixed-anchor/state/grip/missing-arm semantics, recovery/exclusion index, narrow invalid-anchor handling,0.6s DROID metadata windows and explicit gaps,ABC physical anchors,ABC sparse-statistics/phase agreement with analytic values,distributed warmup scaling. This is not GPU acceptance.
