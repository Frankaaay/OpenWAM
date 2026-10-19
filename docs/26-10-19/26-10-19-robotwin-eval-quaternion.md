# RoboTwin native quaternion evaluation correction

User authorized stopping the invalid 16-GPU evaluation, preserving results, publishing the fix, validating official evaluation settings and sampled rollouts, then restarting 16-GPU Clean evaluation. DLC dlcmezbwrojf5dfd and dlcmyyiob115d4ya are confirmed Stopped.

Production training used the published manifest with native_quaternion=wxyz, correctly reordered by the training reader. The evaluation proprio helper incorrectly treated native wxyz as xyzw; the shared action converter also returned xyzw directly to RoboTwin's transforms3d/SAPIEN wxyz planner. Fix only the RoboTwin proprio helper and output adapter boundary. The shared generic xyzw output remains unchanged for other callers.

Previous unit fixtures incorrectly assumed xyzw native input and therefore agreed with the faulty implementation. Replace these fixtures and add physical-pose roundtrips for identity and nontrivial rotations through the actual eval callback, with independent scipy rotation matrices. Real demonstration audit on 32 frames measured 110–180 degree wrong target rotations before the fix. Existing 9/1010 partial successes are contaminated and not valid model scores. Preserve old artifacts; corrected runs must use a new result root and cannot resume these results.

Initial validation: AST parsing and git diff --check pass. Remote CPU contract tests, real demonstration comparison, and sampled GPU rollouts are pending before full evaluation. Training weights and action/statistics definitions are unchanged.
