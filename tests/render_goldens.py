"""Render the golden qpos trajectories to MP4 for visual inspection.

Golden files only prove that behavior did not change; whether the motion is
*right* is certified once, by a human, when the goldens are (re)generated.
Run this script and watch the videos before committing new goldens:

    python tests/render_goldens.py --out_dir /tmp/gmr_goldens
"""

import argparse
import pathlib

import imageio
import mujoco as mj
import numpy as np

from general_motion_retargeting import ROBOT_XML_DICT

from test_retarget_regression import DATA_DIR, ROBOTS

FPS = 30


def render(robot, out_dir):
    qpos = np.load(DATA_DIR / f"golden_qpos_{robot}.npy")
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=480, width=640)
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    camera.distance = 2.5
    camera.azimuth = 135.0
    camera.elevation = -15.0

    out = out_dir / f"golden_{robot}.mp4"
    writer = imageio.get_writer(out, fps=FPS)
    for frame in qpos:
        data.qpos[:] = frame
        mj.mj_forward(model, data)
        camera.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=camera)
        writer.append_data(renderer.render())
    writer.close()
    renderer.close()
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("/tmp"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for robot in ROBOTS:
        render(robot, args.out_dir)


if __name__ == "__main__":
    main()
