"""Robot model loading with optional site injection.

IK configs may declare tracking sites (name, parent body, pos, wxyz quat)
that are added to the robot model at load time. This lets configs target
frames that do not exist in the XML, so stock vendor models can be used
without hand-editing massless dummy bodies into them.
"""

import mujoco as mj


def load_robot_model(xml_path, tracking_sites=None):
    """Load a MuJoCo model, injecting the given tracking sites.

    Args:
        xml_path: Path to the robot MJCF file.
        tracking_sites: Optional mapping of site name to a dict with keys
            "body" (parent body name), "pos" (3 floats, default origin), and
            "quat" (4 floats wxyz, default identity), as read from the
            "tracking_sites" key of an IK config.

    Returns:
        The compiled MjModel with one extra site per entry.

    Raises:
        ValueError: If a site entry is malformed or its parent body does not
            exist in the model.
    """
    if not tracking_sites:
        return mj.MjModel.from_xml_path(str(xml_path))

    spec = mj.MjSpec.from_file(str(xml_path))
    for site_name, entry in tracking_sites.items():
        where = f"tracking site {site_name!r} in {xml_path}"
        if "body" not in entry:
            raise ValueError(f"{where}: missing required key 'body'")
        parent = spec.body(entry["body"])
        if parent is None:
            raise ValueError(
                f"{where}: parent body {entry['body']!r} does not exist "
                "in the robot model"
            )
        pos = entry.get("pos", [0.0, 0.0, 0.0])
        quat = entry.get("quat", [1.0, 0.0, 0.0, 0.0])
        if len(pos) != 3:
            raise ValueError(f"{where}: pos must have 3 values")
        if len(quat) != 4:
            raise ValueError(f"{where}: quat must have 4 values (wxyz)")
        parent.add_site(name=site_name, pos=pos, quat=quat)
    return spec.compile()
