# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation and Setup

This is a Python package for motion retargeting to humanoid robots. Install in development mode:

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
```

## Code Architecture

### Core Components

- **`Retargeter`** (`general_motion_retargeting/retargeter.py`): Stateful differential-IK session built from an explicit robot, profile, and solver settings
- **`KinematicsModel`** (`general_motion_retargeting/kinematics_model.py`): Handles robot kinematics calculations
- **`MotionWorkspace`** (`general_motion_retargeting/visualization.py`): Browser-based mjviser workspace for robot and source motions
- **`Catalog`** (`general_motion_retargeting/catalog.py`): Menagerie-backed robot/profile lookup and external provider discovery

### Data Flow

1. **Human Motion Input**: SMPL-X (AMASS/OMOMO) or BVH (LAFAN1) format
2. **Motion Format**: Each frame = dict of (human_body_name, 3D translation + rotation)
3. **Robot Output**: Tuple of (base_translation, base_rotation, joint_positions)
4. **IK Configs**: JSON files in `general_motion_retargeting/ik_configs/` define human-to-robot body mappings

### Supported Robots

Built-in models come from MuJoCo Menagerie. Use `build_catalog().robots` as
the source of truth. External packages can register additional robots and
profiles through `gmr.providers`.

## Common Commands

### Single Motion Retargeting
```bash
retarget --source <source> --robot <robot_name> --input <path> --output <path>
```

### Visualization
```bash
# Visualize saved robot motion
visualize --robot-motion <motion.npz>
```

Use `stream --source <source> --robot <robot_name>` for live retargeting.

## Key Technical Details

- **IK Solver**: Uses mink library with configurable solver (default: "daqp") and damping (default: 5e-1)
- **Human Height Scaling**: Scales from `HumanMotion.height` and the selected profile assumption
- **Real-time Performance**: Optimized for 60-70 FPS on high-end CPUs for teleoperation use cases
- **Body Model Dependencies**: Resolve SMPL-X body models from an explicit path or `GMR_SMPLX_MODELS`

## File Organization

- `general_motion_retargeting/`: Core library code
- `general_motion_retargeting/cli/`: The three installed command entry points
- `general_motion_retargeting/sources/`: Lazy live-source adapters
- `general_motion_retargeting/ik_configs/`: Versioned named retargeting profiles
- `tests/`: Regression, contract, application, visualization, and streaming tests