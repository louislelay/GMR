# Retargeting profiles

Profiles map named source frames to named robot frames. Version 2 makes both
sides explicit and allows local transforms without editing robot MJCF.

```json
{
  "schema_version": 2,
  "identifier": "smplx_to_my_robot",
  "robot_root_name": "base",
  "human_root_name": "pelvis",
  "ground_height": 0.0,
  "human_height_assumption": 1.8,
  "human_scale_table": {
    "pelvis": 1.0,
    "left_foot": 0.9
  },
  "stages": [
    {
      "root": {
        "robot": {
          "body": "base",
          "position": [0.0, 0.0, 0.0],
          "rotation": [1.0, 0.0, 0.0, 0.0]
        },
        "human": {
          "body": "pelvis",
          "position": [0.0, 0.0, 0.0],
          "rotation": [1.0, 0.0, 0.0, 0.0]
        },
        "position_weight": 100.0,
        "orientation_weight": 10.0
      },
      "left_foot": {
        "robot": {
          "body": "left_ankle",
          "position": [0.1, 0.0, -0.02]
        },
        "human": {
          "body": "left_foot"
        },
        "position_weight": 100.0,
        "orientation_weight": 20.0
      }
    }
  ]
}
```

Positions are XYZ meters. Rotations are unit WXYZ quaternions. Omitted frame
positions and rotations default to zero and identity.

Each stage is solved in order. A match with both weights at zero is inactive.
Every active human body must appear in `human_scale_table`. Scales and
`human_height_assumption` must be positive and finite.

Robot transforms are implemented as generated MuJoCo sites. Site names are
internal and collision-free; the profile identifies the stable parent body.
Human transforms are applied independently for each stage. Profiles without
`schema_version: 2` are rejected.

## Validation workflow

1. Load the profile with `load_profile`.
2. Construct `Retargeter`; this validates robot bodies and compiles generated
   sites.
3. Retarget representative motions.
4. Compare named-joint trajectories and inspect source/robot sites in
   `visualize`.
5. Commit intentional golden changes with the profile.

Validation errors include the profile field, missing source body, or missing
robot body. Do not work around errors by adding empty bodies to vendor MJCF;
put the intended transform in the profile.
