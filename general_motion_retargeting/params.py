import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",
    "unitree_g1_with_hands": ASSET_ROOT
    / "unitree_g1"
    / "g1_mocap_29dof_with_hands.xml",
    "unitree_h1": ASSET_ROOT / "unitree_h1" / "h1.xml",
    "booster_t1": ASSET_ROOT / "booster_t1" / "T1_serial.xml",
    "fourier_n1": ASSET_ROOT / "fourier_n1" / "n1_mocap.xml",
    "pnd_adam_lite": ASSET_ROOT / "pnd_adam_lite" / "scene.xml",
    "pal_talos": ASSET_ROOT / "pal_talos" / "talos.xml",
}

IK_CONFIG_DICT = {
    # offline data
    "smplx": {
        "unitree_g1": IK_CONFIG_ROOT / "smplx_to_g1.json",
        "unitree_g1_with_hands": IK_CONFIG_ROOT / "smplx_to_g1.json",
        "unitree_h1": IK_CONFIG_ROOT / "smplx_to_h1.json",
        "booster_t1": IK_CONFIG_ROOT / "smplx_to_t1.json",
        "fourier_n1": IK_CONFIG_ROOT / "smplx_to_n1.json",
        "pnd_adam_lite": IK_CONFIG_ROOT / "smplx_to_adam.json",
    },
    "bvh_lafan1": {
        "unitree_g1": IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json",
        "unitree_g1_with_hands": IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json",
        "fourier_n1": IK_CONFIG_ROOT / "bvh_lafan1_to_n1.json",
        "pal_talos": IK_CONFIG_ROOT / "bvh_to_talos.json",
    },
    "bvh_nokov": {
        "unitree_g1": IK_CONFIG_ROOT / "bvh_nokov_to_g1.json",
    },
    "bvh_xsens": {
        "unitree_g1": IK_CONFIG_ROOT / "bvh_xsens_to_g1.json",
    },
    "fbx": {
        "unitree_g1": IK_CONFIG_ROOT / "fbx_to_g1.json",
        "unitree_g1_with_hands": IK_CONFIG_ROOT / "fbx_to_g1.json",
    },
    "fbx_offline": {
        "unitree_g1": IK_CONFIG_ROOT / "fbx_offline_to_g1.json",
    },
    "xrobot": {
        "unitree_g1": IK_CONFIG_ROOT / "xrobot_to_g1.json",
    },
    "xsens_mvn": {
        "unitree_g1": IK_CONFIG_ROOT / "xsens_mvn_to_g1.json",
    },
}


ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
    "unitree_g1_with_hands": "pelvis",
    "unitree_h1": "pelvis",
    "booster_t1": "Waist",
    "fourier_n1": "base_link",
    "pnd_adam_lite": "pelvis",
    "pal_talos": "base_link",
}

VIEWER_CAM_DISTANCE_DICT = {
    "unitree_g1": 2.0,
    "unitree_g1_with_hands": 2.0,
    "unitree_h1": 3.0,
    "booster_t1": 2.0,
    "fourier_n1": 2.0,
    "pnd_adam_lite": 3.0,
    "pal_talos": 3.0,
}
