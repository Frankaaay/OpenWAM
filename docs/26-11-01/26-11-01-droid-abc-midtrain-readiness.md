# DROID + ABC stereo midtrain readiness

2026-11-01, Asia/Shanghai.

## Confirmed scope

- Wan-initialized video DiT, newly initialized ActionDiT; frozen Stage3 u8000 StereoTok, not the earlier Stage2 u8000 export. Intended downstream is a real robot.
- One complete epoch over all valid training windows, with episode-isolated validation. Eight fixed GPUs, no idle-resource substitution. No formal training submitted in this investigation.
- User explicitly requested W&B offline, overriding the repository online default for this task. Persist W&B offline files, console logs, resolved config, normalization, validation JSONL, and full recovery state on personal CPFS.
- Necessary data adapters and periodic validation changes are authorized. No branch, worktree, commit, push, data transfer, or deletion was performed.

## Source and local change

OpenWAM branch `frank/eighteen-preparation`, HEAD `4a06f9a3c78a40605fb3a4f56d60dffb1251c220`. Both origin and frank fetch succeeded; HEAD matches `frank/frank/eighteen-preparation`. Preexisting changes in the LeRobot reader, LIBERO test, and older documents are preserved. No pull in the dirty worktree.

Existing `training.validation.every_steps` already validates at start, each optimizer-step interval, and completion. It assumed the first RoboTwin sub-dataset. The local change extends that method to named mixtures and LeRobot episode indexes, checks overlap across all included leaves, keeps four fixed windows per held-out episode, and emits aggregate plus per-source loss/action_loss/video_loss/window counts. Equal action dimensions are required; no implicit validation-only padding semantics are introduced. Existing RNG/module-mode restoration and distributed reduction remain in use. W&B initialization already accepts `WANDB_MODE=offline`; the validation error text now accurately permits either online or offline runs.

Validation: `C:/Project/StereoTok/.venv/Scripts/python.exe C:/Project/OpenWAM/tests/test_midtrain_validation.py` passed four CPU tests of the actual method isolated through AST loading: offline metric/JSONL output and repeatability/RNG/mode restoration, second-source overlap rejection, nonfinite rejection, and legacy RoboTwin episode-file compatibility. `git diff --check` passed. No pytest installed in that local environment. No full-model or distributed GPU validation claimed.

## Live cloud evidence

RAM identity `frank`, user `205455090823393499`, workspace `350607` verified. Research quota `quotan2i9e2t9tqr` is Ready and accessible; API GPU type is L20Y. ActualMinQuota GPU32, AllocatedQuota GPU32, EmptyNodeNum0; these fields do not prove eight immediately schedulable fixed GPUs. No DSW exists for this user.

Read-only inventory jobs use 0GPU/2CPU/32Gi, priority4, `ForbiddenQuotaOverSold`, maximum15 minutes, RO data and CPFS mounts:

- `dlcnffdord1u2vzf`: inventory reached `MIDTRAIN_INVENTORY_DONE`.
- `dlc1he7j0rkmt31s`: metadata audit Succeeded, `MIDTRAIN_METADATA_DONE`.
- `dlc2mb8jvw4krz1n`: Succeeded, `ABC_COUNT_DONE`; bounded ABC episode metadata count and representative stereo decode completed.

Exact scripts, request JSON, receipts, and collected logs: `C:/Users/Frank/AppData/Local/AliyunCLI/h800-requests/openwam-midtrain-20261101/`. Commands run through native Aliyun CLI profile h800, region cn-shenzhen. An initial log query used unsupported `--Tail`; local CLI rejected it before a log request. Corrected to `--MaxLines`, using actual master pod IDs. No remote failure inferred from that CLI error.

## Data findings

Stage3 checkpoint `/mnt/workdir/frank/runs/stage3-u8000-wan-eval-20261028/input/stage3-u8000.ckpt` exists, 13,799,251,677 bytes. This audit checked existence/size, not a fresh full hash or weight load.

DROID root `/mnt/workdir/frank/datasets/tokenizer-h100-20261019/droid-prepared-20260920/` exists. Live-read formal summary declares 72,557 episodes, 805,355 train windows, 41,846 val windows and 44,836 test windows, with lab/scene split and `full_corpus_audited=false`. These are tokenizer windows, not final policy window counts. Three episode directories contain Lance stereo frames and timing/geometry metadata, but no local action HDF5. Their result.json references `/gpfs/jiuquyun/datasets/DROID/raw/1.0.1/<episode>/trajectory.h5`. Exact corresponding probes under the known H800 personal/shared roots all returned missing. This is bounded evidence, not an exhaustive cloud-wide absence proof. No action migration is considered complete.

ABC root `/mnt/data/data/h100-jiuquyun/PRETRAIN_DATA/ABC-130k-yam-canonical-v1/`: actual info.json declares 100,028 episodes, 267,756,054 frames, 30Hz, bimanual20D achieved state, no stored action. Parquet schema and one state row independently confirm this. Do not mix achieved-next-state labels with DROID commanded wrist-pose labels without an explicit target convention.

Base mount's meta/episodes is empty. Reused the established RO overlays from `/peilin/StereoTok/metadata/abc-full-20261024/episodes/` and `/peilin/StereoTok/metadata/abc-sidecars-20261024/episode_sidecars/`; episode Parquet then reads successfully. These shared metadata inputs are read-only; no code/environment or Git source from another user is used. One sampled episode has `cam_head_right=false`: the whole 100,028-episode publication is not the stereo population.

Sequential scan of the101 episode Parquet files (metadata columns only): 11,936 episodes/37,333,742 frames have both head views; 88,092 episodes/230,422,312 frames do not. Whole-publication source_split labels are98,792 train/1,236 val; stereo-only split counts have not been computed. Episode0 left/right MP4 first frames both decode to256x256. This is representative decode evidence, not full-corpus integrity or timing validation. The referenced episode0 sidecar path did not resolve as a directory; sidecar storage/path mapping remains to be checked if required by the adapter.

Actual OpenWAM environment has torch, pyarrow, h5py, wandb and av, but no Lance. The existing stage3-eval environment has Lance and h5py but no wandb. Reusing the raw Lance payload therefore requires a compatible Lance dependency in the training environment (or an explicitly selected data conversion); no environment was modified during this audit.

Recommended next data step: retain existing image payloads; inspect and provision only missing DROID trajectory/action/state/timestamp data with UUID/episode joins, then implement the reader against the agreed target/frequency contract. Do not recopy SVOs, silently use mono episodes, silently drop unreadable samples, or replace all-window coverage with weighted resampling. Existing proportional MixtureDataset supports one visit per source window before DataLoader tail handling; final coverage/recovery verification remains required.

## Pending semantic decisions and gates

Need real-robot embodiment, commanded interface (joints versus EEF pose/delta), coordinate frames, control rate, and deployed camera setup. Recommend a common EEF pose/gripper representation with explicit missing-arm masks for cross-embodiment midtrain; ABC can supply future achieved-pose targets, which are not recorded commands. Final target conversion and DROID timestamp matching await those decisions and actual HDF5 availability.

Proposed validation cadence is every500 optimizer steps plus start/end, with a fixed episode-isolated subset per source. This is a proposed recipe, not a launched run/config. Full readers, normalization, one-epoch sample count, memory/throughput, eight-GPU validation, and ETA remain unverified. Access to H100 and any transfer requires a scoped follow-up authorization; current investigation used only H800.

## Follow-up: authorized H100 read-only investigation

User clarified the downstream robot is bimanual and requested investigating H100 instead. H100 read-only access is now authorized; training submission, dependency installation and transfer were not performed. Read both Frank and shared H100 guides; SSH config and remote whoami agree on Frank. All remote probes used ssh.exe with simple single-quoted commands or LF-normalized Python stdin; no remote source edits.

Live evidence: DROID original trajectory.h5 exists for RPL/success/2023-06-04/Sun_Jun__4_12:26:16_2023 (1,341,400 bytes) and TRI/success/2023-12-04/Mon_Dec__4_15:00:01_2023 (2,869,506 bytes), under /gpfs/jiuquyun/datasets/DROID/raw/1.0.1. Prepared stereo root and formal summary also exist at their original Frank paths. ABC info.json, episode Parquet directory, data directory and head-left video directory exist under /gpfs/jiuquyun/datasets/PRETRAIN_DATA/ABC-130k-yam-canonical-v1. These are existence/metadata checks, not full-corpus action integrity or timestamp validation. Attempt to inspect HDF5 schema with the existing StereoTok interpreter stopped at missing h5py; no schema result is claimed.

Existing StereoTok environment has lance/pyarrow/wandb/av but lacks h5py/accelerate/deepspeed/diffsynth. No /gpfs/jiuquyun/projects/Frank/OpenWAM or Frank/envs exists; bounded personal-project checks did not locate a ready OpenWAM runtime. Known Wan snapshot contains only Wan2.2_VAE.pth and its SHA file. Full Wan DiT/T5 and the exact Stage3 u8000 weights on H100 remain unlocated; similarly named older u8000 outputs are not substituted. Shared GPFS reports301T total/290T used/11T free: shared capacity, not a personal quota or reserved space.

Slurm account compute-account permits normal/debug/long/cpu, plus idle-compute/idle. Normal MaxWall2days, long7days. Nodes each declare8GPU; several were IDLE+PLANNED, not immediately available. `sbatch --test-only --account=compute-account --partition=h100 --qos=long --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=32 --mem=768G --time=7-00:00:00 --wrap=true` succeeded, predicted2026-11-04T05:16:00 on xn01-gpu1-0060 in h100-8. This is only a scheduling estimate; no training job was submitted (the displayed test-only identifier31828 is not a launched job). Subsequent squeue for Frank still lists only30575. Both inspected partitions report PreemptMode=CANCEL; fixed GPU count does not imply immunity from preemption. Existing job30575 was untouched.

Clarification of supervision: current RoboTwin EEF reader uses raw_actions[0:1] as proprio and raw_actions[1:num_frames] as targets, with endpose xyz/rotation and gripper. Current OxeDroidDataset instead uses commanded other_information.action_wrist_pose, with achieved observation_gripper_pose6d as proprio. ABC state-only conversion still needs a defined temporal rule. The earlier suggestion to change DROID to future achieved-state targets is optional, not a readiness requirement; preserve existing source semantics unless the user explicitly chooses otherwise. Real-robot exact control interface may be settled during downstream fine-tuning; midtrain still requires coherent slots, units, masks, normalization and temporal alignment.

DROID ACTION_DIM_MASK=LEFT_ARM_DIM_MASK; unify mapping places the valid10D arm in slots0:10 and the absent arm in34:44, with other unused dimensions invalid. Reader builds action/proprio dimension masks, architecture converts action_mask to action_is_pad, and per-element loss divides by the count of valid elements. DROID missing-arm masking applies per sample in both training and validation, not globally to ABC/bimanual data. It excludes direct missing-arm supervision; shared network parameters can still update. Visual camera availability is independent of arm-action availability.

## 2026-11-01 authorized preparation and current blocker

Later user instructions supersede proposals above: H100 16 GPUs (two nodes), one short 16-GPU smoke then formal one epoch; W&B offline; frozen Stage3 u8000 and T5; Wan Video DiT initialization and random Action DiT. Both sources use observed state targets and Orion fixed-anchor transforms/interpolation/normalization. Robot controller details are deferred to finetuning. Environment installation, weight downloads, necessary asset migration and reader adaptation are authorized.

CPU asset job31927 prioritizes Stage3 before environment and project transfers. Live log contains STAGE3_TRANSFER_DONE; destination `/gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101/mirror/mnt/workdir/frank/runs/stage3-u8000-wan-eval-20261028/input/stage3-u8000.ckpt` is13,799,251,677 bytes, matching source. This is successful rsync completion and size verification, not a new full SHA verification or GPU weight load. Environment transfer remained in progress at the first check; no formal training has started.

CPU audit31937 streams the existing formal manifest and checks original DROID HDF5 state fields. It enumerated72,457 unique represented episodes and805,355 train/41,846 val/44,836 test windows. This differs from the earlier summary's72,557 total; use actual manifest membership for policy preparation. First5,000 audits had4 missing trajectory_h5 metadata fields; no samples were excluded or shared files modified. Audit remains running and is not a full-corpus result.

Targeted source investigation confirmed two IRIS episodes have no trajectory object in the official GCS publication either: `IRIS/success/2023-07-06/Thu_Jul__6_16:34:30_2023` and `IRIS/success/2023-11-02/Thu_Nov__2_16:01:49_2023`. Exact-prefix GCS JSON API listings returned no objects for each episode's trajectory prefix. These cannot currently provide observed-action supervision. A third example, `ILIAD/success/2023-07-19/Wed_Jul_19_20:38:19_2023`, has a published trajectory_im128.h5 (129,246,219 bytes) and needs targeted retrieval/schema inspection rather than exclusion. No substitute commanded targets or fabricated trajectories were used. Excluding truly unsupervised episodes changes the admitted population and awaits explicit confirmation under AGENTS.md.

The local Orion adapter draft is not yet registered, indexed, normalized or runtime validated. Removed an unused dead expression; py_compile and git diff --check passed. Source fetch initially failed with a local TLS handshake and succeeded on retry; baseline remains4a06f9a3c78a40605fb3a4f56d60dffb1251c220 with preexisting unrelated modifications preserved. No commit/push, 16-GPU smoke or formal launch is claimed. Diagnostic scripts and receipts remain in the existing local AliyunCLI openwam-midtrain-20261101 directory and H100 preparation root.

## Authorized missing-state handling

User explicitly approved listing and excluding only samples whose observed state cannot be recovered, while retaining recoverable episodes and training all remaining valid windows once. This resolves the missing-state exclusion decision above; it does not authorize dropping other schema/timing errors or substituting commanded actions.

Submitted CPU recovery job32093 (2CPU/8GiB, two-hour limit) after bash syntax and Slurm test-only checks. It joins current audit failures to the original manifest, queries the exact official trajectory prefix, prefers trajectory.h5 then trajectory_im128.h5, downloads only recoverable assets into the private preparation root's `datasets/droid-state-supplement/`, verifies published size/MD5 and observed state schema/timestamps/gripper range, and appends provenance to `droid-state-recovery.jsonl`. Shared source data is not modified. A successful-start log already records the two confirmed IRIS missing-state exclusions; recovery of the ILIAD alternative was still in progress at that check.

Job32094 has dependency `afterok:31937:32093`, so after the full audit and initial recovery succeed it processes newly discovered errors once, preserving earlier receipts. Jobs31927 (remaining environment/assets) and31937 (full action schema audit) continue. These are preparation jobs, not a training launch. Recovery script/sbatch are in the existing local operational-artifact directory and copied to `/gpfs/jiuquyun/projects/Frank/openwam-midtrain-20261101/`; log pattern `/gpfs/jiuquyun/home/Frank/logs/openwam-droid-state-recovery-%j.out`.

## Persistent launch goal and implementation

User requested continuous autonomous work until actual training starts, selecting recommended defaults for intervening issues. Initial recovery32093 finished: three recovered official trajectory_im128 files passed published size/MD5 and HDF5 state checks; three unrecoverable episodes are explicitly listed. Full audit remains active (9,000 examined,7 errors at latest bounded check).

Added task-specific Orion readers, lazy dataset registration, SQLite indexing with explicit exclusions/recovered paths, and exact training-population normalization using Orion RunningMoments/compute_zscore_statistics. ABC preserves published episode splits and requires real head-left/right; DROID maps its three true stereo cameras to the existing canvas. Both use17 RGB frames10Hz and34 action targets20Hz at Orion offsets -0.05 through1.6s, a single current-observation anchor, and the existing80D slots. DROID's recorded EE frame remains its frame; no invented TCP offset. Actual state timestamps use host read midpoint, not the robot's differently offset clock.

Recipe uses16 GPUs, microbatch2/rank, accumulation4 (global128), Video DiT LR3e-6 and random Action DiT LR1e-4, standard cosine schedule, one epoch, no extra mixture balancing. Offline validation uses a deterministic cap of128 windows/source, every500 optimizer steps plus initial/final. Full state is now also saved at a final off-cadence step; high retention prevents automated pruning for this run. Final launcher still needs real-data/16-GPU validation. No new training result is claimed.

Seven local CPU tests passed: two physical target/admission tests and five validation tests (including per-source limit). Preparation CLI imports and --help succeeded using isolated local test dependencies; no shared local environment changed. Initial Windows test cleanup exposed an unclosed SQLite test handle, fixed with contextlib.closing; readers likewise close their construction connection. These tests do not substitute for GPU smoke. Required source synchronization uses the existing personal OpenWAM branch with only this task's named files; preexisting unrelated changes are retained separately.

Published OpenWAM70a3bc6a9e7835f507ac36cf8c4719c94adf1c30 to personal frank/eighteen-preparation and created own clean H100 clone. Orion pinned783b602d128a43814af32285e60a60f270ec2703 and StereoTok7dae078f7509b2e9c7c69ed8162a62263a99a077 are separate own clean clones. Orion's old stereotok-pr remote ref was removed: pull failed; no local branch update or new Orion code was used. The pinned published commit remains an ancestor of current origin/main (verified), and is the same implementation tested locally.

ABC CPU job32124 installed isolated preparation dependencies and exposed a dataset identifier mismatch: existing mask metadata says abc130k, not abc. Fixed only the adapter's source name; no metadata or semantics changed. DROID threaded audit31937 was cancelled intentionally after preserving its result file; resume job32126 uses8 processes to avoid h5py's global in-process lock and carries forward complete prior rows into droid-action-audit-complete.jsonl. Final summary identifies its audit_path. Recovery32094 now depends on32126; an attempted dependency including the already-completed32093 was rejected by Slurm, then corrected to the new audit alone. No source data deletion or silent exclusion occurred.
