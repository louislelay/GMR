"""Offscreen (headless) video rendering of robot motions.

Renders qpos trajectories to a video file with MuJoCo's offscreen renderer,
so motions can be inspected on machines without a display (servers, CI).
Requires a headless-capable GL backend; set MUJOCO_GL=egl or MUJOCO_GL=osmesa
when no display is available.
"""

import imageio
import mujoco as mj
import numpy as np

from .params import ROBOT_XML_DICT


def render_robot_motion(
    robot_type: str,
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    fps: float,
    video_path,
    width: int = 640,
    height: int = 480,
    camera_distance: float = 2.5,
    camera_azimuth: float = 135.0,
    camera_elevation: float = -15.0,
) -> None:
    """Render a robot qpos trajectory to a video file, without a window.

    The camera tracks the floating base at a fixed distance and angle.

    Args:
        robot_type: Robot key in ROBOT_XML_DICT (e.g. "unitree_g1").
        root_pos: (num_frames, 3) floating-base positions.
        root_rot_wxyz: (num_frames, 4) scalar-first base quaternions.
        dof_pos: (num_frames, num_joints) joint positions.
        fps: Playback frame rate of the output video.
        video_path: Output file; the container is inferred from the suffix.
        width: Frame width in pixels.
        height: Frame height in pixels.
        camera_distance: Distance from the camera to the robot base.
        camera_azimuth: Camera azimuth in degrees.
        camera_elevation: Camera elevation in degrees.
    """
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot_type]))
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=height, width=width)
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    camera.distance = camera_distance
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation

    writer = imageio.get_writer(video_path, fps=fps)
    try:
        for frame_idx in range(len(root_pos)):
            data.qpos[:3] = root_pos[frame_idx]
            data.qpos[3:7] = root_rot_wxyz[frame_idx]
            data.qpos[7:] = dof_pos[frame_idx]
            mj.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
