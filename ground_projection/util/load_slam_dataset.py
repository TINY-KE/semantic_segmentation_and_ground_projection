import os

def load_pairs_from_association_file(file_path, imgs_root_path, valid_timestamps):
    """
    从关联文件中加载符合条件的 RGB 和深度图路径对。

    参数:
        file_path (str): 关联文件路径
        imgs_root_path (str): 图像根目录
        valid_timestamps (set or list): 有效的时间戳集合

    返回:
        list_depth: [{'timestamp': ..., 'fpath_img': ...}, ...]
        list_rgb:   [{'timestamp': ..., 'fpath_img': ...}, ...]
    """
    print(f"Loading pairs from association file: {file_path}")

    list_rgbd = []

    # 确保 valid_timestamps 是 set，提高查找效率
    if not isinstance(valid_timestamps, set):
        valid_timestamps = set(valid_timestamps)

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):

            parts = line.strip().split()
            if len(parts) < 4:
                print(f"[Line {line_num}] 跳过格式错误的行: {line.strip()}")
                continue

            timestamp, depth_path, _, rgb_path = parts[:4]
            timestamp = int(timestamp)
            if timestamp in valid_timestamps:
                # 构造完整路径
                depth_full_path = os.path.join(imgs_root_path, depth_path)
                rgb_full_path = os.path.join(imgs_root_path, rgb_path)

                list_rgbd.append({
                    'timestamp': timestamp,
                    'fpath_depth': depth_full_path,
                    'fpath_rgb': rgb_full_path
                })
            else:
                print(f"[Warning] timestamp {timestamp} 不在 valid_timestamps 中")
                print(f"valid_timestamps 中的例子: {list(valid_timestamps)[:5]}")  # 只显示前5个

    print(f"✅ 成功加载 {len(list_rgbd)} 对图像")
    return list_rgbd

def load_poses_from_file(poses_file, skip_every_n=1):
    """
    从 KeyFrames_for_smp.txt 文件中读取位姿数据。

    每行格式: time x y z qx qy qz qw
    返回:
        poses: List[Tuple[int, List[float]]]，每个元素是 (time, [x, y, z, qx, qy, qz, qw])
    """
    poses = []
    valid_timestamps = []

    try:
        with open(poses_file, 'r') as f:
            for idx, line in enumerate(f):
                if idx % skip_every_n != 0:
                    continue  # 跳过不需要的行（每隔 N 行保留一行）

                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # 跳过空行和注释

                parts = line.split()
                if len(parts) != 8:
                    print(f"[warning] 跳过格式错误的行: {line}")
                    continue

                try:
                    time = int(parts[0])
                    pose = list(map(float, parts[1:]))
                    # translation = pose[:3]  # 位移: [x, y, z]
                    # quaternion = pose[3:]  # 四元数: [qx, qy, qz, qw]
                    valid_timestamps.append(time)
                    # poses.append((translation, quaternion))
                    poses.append(pose)
                except ValueError:
                    print(f"[error] 无法解析该行: {line}")

    except FileNotFoundError:
        print(f"[error] 文件未找到: {poses_file}")

    return valid_timestamps, poses