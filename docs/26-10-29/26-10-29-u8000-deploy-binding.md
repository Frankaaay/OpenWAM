# u8000 deployment binding repair

Purpose: unblock the authorized 16-GPU idle-resource Clean-only evaluation of the final step-30000 OpenWAM StereoTok u8000 checkpoint.

Base: frank/eighteen-preparation, ac3f318eeef414a5d40c6df820e68f57f40845fa. Only encoder/stereotok.py from_skeleton changes: admit the exact known u8000 SHA256 alongside u15853. Training initialization, architecture checks, input-mode checks and strict checkpoint tensor loading remain unchanged. The current right-eye inference path is retained; the older c3be7d0 u8000 training checkout lacks the subsequent engine right_first_frame_image forwarding and must not replace the full evaluation source.

The shared-robotwin-20261023 OpenWAM directory resolves to the 20261021 checkout, clean HEAD 11abfa5bf382c04457ae9824c48fb472ec502527. Its u15853-only guard reproduces the failure. The final u8000 checkpoint directory is /mnt/workdir/frank/runs/openwam-robotwin-stereotok-u8000-20261023/checkpoints/2026-10-24_10-08-39.

Validation: local syntax compilation and targeted diff check pass. CPU/meta diagnostic dlc1x7bxmzd31572 reproduced the original rejection, built the patched skeleton using real checkpoint metadata (769 state keys, stereo required), and rejected an unknown hash. The negative disparity case was correctly rejected but the test expected the wrong error text; test-only correction submitted as dlcnz7itnls1bvmf. This is not GPU weight-load or episode evidence.

User approved committing and pushing this patch and record, followed by the original 16-GPU idle-resource Clean-only evaluation. Publish only these two files. Formal runtime will bind the resulting exact SHA in a separate clean clone; unrelated running checkouts remain unchanged.

Validation completion: dlcnz7itnls1bvmf printed GUARD_TEST_DONE. Actual u8000 metadata skeleton and u15853 identity regression passed; unknown hash, invalid disparity and input-mode mismatch were rejected. GPU checkpoint loading and rollout remain unverified until authorized publication and launch.
