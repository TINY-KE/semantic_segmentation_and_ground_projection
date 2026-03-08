# camera_intrinsics.py

camera_intrinsics = {
    "TUM_RGBD": {
        "width": 640,
        "height": 480,
        "fx": 481.20,
        "fy": -480.00,
        "cx": 319.50,
        "cy": 239.50,
        "depth_factor": 5000.0
    },
    "KINECT_DK": {
        "width": 1280,
        "height": 720,
        "fx": 614.918291,
        "fy": 617.128471,
        "cx": 635.265398,
        "cy": 377.167164,
        "depth_factor": 5000.0
    }
}



def get_intrinsics(camera_type="TUM_RGBD"):
    if camera_type not in camera_intrinsics:
        raise ValueError(f"未找到相机类型: {camera_type}")
    p = camera_intrinsics[camera_type]
    return (p["width"], p["height"], p["fx"], p["fy"], p["cx"], p["cy"], p["depth_factor"])