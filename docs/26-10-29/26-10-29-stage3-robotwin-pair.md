# Stage3 RoboTwin mono / stereo pair

2026-10-29 Asia/Shanghai. User authorized two fresh 8-GPU idle-resource jobs, retaining the u15853 Wan-init RoboTwin training contract: clean_50, seed42, 30000 updates, batch16/rank, accumulation1, 8 ranks, original data/normalization/LR/loss. Only tokenizer checkpoint, input mode and independent run/output identity change.

The supplied Stage3 training checkpoint is read by the strict FastWAM-C2R vendor loader. Remove the old u15853 checksum allowlist while preserving architecture checks, recorded checkpoint SHA256/input-mode binding and strict policy weight loading. Read geometry from training metadata or the established deployment export. Training-only semantic/alignment heads are excluded by the vendor; RGB/fusion/geometry weights are retained.

Existing unrelated reader/test changes and old documents are preserved and excluded from this publication. Local AST/diff checks precede remote exact-SHA import, checkpoint load, mono/stereo input and W&B-online validation. Source, job IDs and runtime results are recorded in StereoTok docs/26-10-29/26-10-29-stage3-robotwin-four.md.
