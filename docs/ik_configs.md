# Authoring and debugging IK configs

An IK config is a JSON file that maps human motion frames to robot tracking
targets. Configs live in `general_motion_retargeting/ik_configs/` and are
registered per (source format, robot) pair in
`general_motion_retargeting/params.py` (`IK_CONFIG_DICT`).

Configs are validated when `GeneralMotionRetargeting` is constructed; any
structural problem (unknown body, bad weight, missing scale entry, ...) raises
a `ValueError` naming the offending entry instead of failing mid-motion.

## File structure

```jsonc
{
    "robot_root_name": "pelvis",          // floating-base body of the robot
    "human_root_name": "pelvis",          // root body name in the human data
    "ground_height": 0.0,                 // subtracted from all pos offsets (z)
    "human_height_assumption": 1.8,       // height the scale table was tuned for
    "use_ik_match_table1": true,
    "use_ik_match_table2": true,
    "human_scale_table": { ... },         // human body -> scale factor
    "ik_match_table1": { ... },           // stage-1 tracking tasks
    "ik_match_table2": { ... },           // stage-2 refinement tasks
    "tracking_sites": { ... }             // optional, injected model sites
}
```

## Match tables

Each entry maps a **robot frame** (the key) to a task tracking a **human
body**:

```jsonc
"left_wrist_yaw_link": [
    "left_wrist",          // human body whose pose is the target
    50,                    // position weight (0 disables position tracking)
    5,                     // orientation weight (0 disables orientation)
    [0.0, 0.0, 0.0],       // position offset, in the (rotated) local frame
    [0.5, 0.5, 0.5, 0.5]   // rotation offset, wxyz quaternion
]
```

- The key must name a **body or site** of the robot model. Entries with both
  weights zero are ignored.
- The rotation offset is applied on the *human* side: the target orientation
  is `human_quat * rot_offset`. Tune it so the robot link's local axes line up
  with the human body's convention (a wrong offset shows up as a permanently
  twisted link).
- The position offset is expressed in the frame *after* the rotation offset is
  applied, then added to the human body position.

### Two solve stages

Retargeting solves `ik_match_table1` first, then `ik_match_table2` on the
resulting configuration. The common pattern is identical tables with different
weights: stage 1 places the pelvis and feet (weights around 100), stage 2
refines end effectors. Low-weight entries (around 10) act as posture guidance
and are *not* expected to converge to zero error when human and robot limb
proportions differ.

## Scaling

`human_scale_table` scales each human body's position toward the human root
before targets are set, adapting human proportions to the robot's. Two rules:

- the human root must have a scale entry;
- every body in the scale table must also appear as the human body of a
  weighted `ik_match_table1` entry (its offsets are looked up there).

`human_height_assumption` documents the subject height the scale table was
tuned for. When a different `actual_human_height` is passed to
`GeneralMotionRetargeting`, all scale factors are multiplied by
`actual_human_height / human_height_assumption`.

## Tracking sites

If the robot XML has no convenient frame to track (e.g. a toe point or a head
target), declare it under `tracking_sites` instead of hand-editing dummy
bodies into the vendor XML:

```jsonc
"tracking_sites": {
    "left_toe_site": {
        "body": "left_ankle_roll_link",   // existing parent body
        "pos": [0.1, 0.0, -0.02],         // offset in the parent frame
        "quat": [1.0, 0.0, 0.0, 0.0]      // optional, wxyz, default identity
    }
}
```

The sites are injected into the model at load time and can be used as match
table keys like any body.

## Debugging workflow

1. **Construct early.** `GeneralMotionRetargeting("smplx", "<robot>")` runs
   full validation; fix `ValueError`s until it constructs.
2. **Retarget a short clip and render it.** On a headless machine:
   `MUJOCO_GL=egl uv run scripts/vis_robot_motion.py --robot <robot>
   --robot_motion_path out.npz --offscreen --video_path out.mp4`.
3. **Read the failure mode:**
   - *Robot floats or sinks*: wrong `ground_height` or pelvis position offset.
   - *A link is permanently twisted*: wrong rotation offset for that entry
     (remember wxyz order, scalar first).
   - *Limbs lag or undershoot*: weights too low relative to stage-1 tasks, or
     scale factors shrink the target out of reach.
   - *Feet slide*: feet weights too low versus pelvis weight.
4. **Pin the result.** `tests/` contains golden-trajectory regression tests;
   after an intentional tuning change, regenerate the goldens with
   `GMR_REGEN_GOLDENS=1 uv run pytest tests/test_retarget_regression.py`,
   then attach a rendered video of the new motion to the pull request.
