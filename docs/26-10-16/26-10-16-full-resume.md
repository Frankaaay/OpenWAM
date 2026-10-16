# Full-state recovery

Authorized pre-training recovery experiment. Reject incomplete checkpoints before loading; retain full states when enabled, without automatic pruning/deletion. Save only at completed optimizer boundaries; preserve actual optimizer counter across epoch tails. Recreate seeded epoch loaders and replay consumed batches to advance worker prompt/augmentation RNG. Production parameter budgets unchanged. GPU checkpoint experiment pending. No formal training launched.
