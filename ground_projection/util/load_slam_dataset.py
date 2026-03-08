import os

# def load_pairs_from_association_file(file_path, imgs_root_path, valid_timestamps, valid_poses, order_depth_rgb = True, time_threshold = 0.05):
#     """
#     从关联文件中加载符合条件的 RGB 和深度图路径对。
#
#     参数:
#         file_path (str): 关联文件路径
#         imgs_root_path (str): 图像根目录
#         valid_timestamps (set or list): 有效的时间戳集合
#
#     返回:
#         list_depth: [{'timestamp': ..., 'fpath_img': ...}, ...]
#         list_rgb:   [{'timestamp': ..., 'fpath_img': ...}, ...]
#     """
#     print(f"Loading pairs from association file: {file_path}")
#
#     list_rgbd = []
#
#     # 确保 valid_timestamps 是 set，提高查找效率
#     if not isinstance(valid_timestamps, set):
#         valid_timestamps = set(valid_timestamps)
#
#     with open(file_path, 'r') as f:
#         for line_num, line in enumerate(f, start=1):
#
#             parts = line.strip().split()
#             if len(parts) < 4:
#                 print(f"[Line {line_num}] 跳过格式错误的行: {line.strip()}")
#                 continue
#
#             if order_depth_rgb:
#                 timestamp, depth_path, _, rgb_path = parts[:4]
#             else:
#                 timestamp, rgb_path, _, depth_path = parts[:4]
#
#             timestamp = float(timestamp)
#
#             # 打印读取结果
#             # print(f"[{'Depth-First' if order_depth_rgb else 'RGB-First'}] "
#             #       f"Time: {timestamp:.4f} | RGB: {rgb_path} | Depth: {depth_path}")
#
#             # 从 valid_timestamps 里查找第一个与 timestamp 的差值小于 0.01 的时间戳。我们逐步解释一下
#             matched_ts = next((ts for ts in valid_timestamps if abs(ts - timestamp) < time_threshold), None)
#             if matched_ts is not None:
#                 # 计算当前的差值
#                 diff = abs(matched_ts - timestamp)
#
#                 # 打印匹配信息，保留 6 位小数以观察细微差异
#                 # print(f"✅ Match Found: Original({timestamp:.4f}) -> Matched({matched_ts:.4f}) | Diff: {diff:.6f}")
#
#                 # 构造完整路径
#                 depth_full_path = os.path.join(imgs_root_path, depth_path)
#                 rgb_full_path = os.path.join(imgs_root_path, rgb_path)
#
#                 list_rgbd.append({
#                     'timestamp': matched_ts,
#                     'fpath_depth': depth_full_path,
#                     'fpath_rgb': rgb_full_path
#                 })
#
#             # else:
#                 # print(f"[Warning] timestamp {timestamp} 不在 valid_timestamps 中")
#                 # print(f"valid_timestamps 中的例子: {list(valid_timestamps)[:5]}")  # 只显示前5个
#
#     print(f"✅ 成功加载图像 {len(list_rgbd)} 对")
#     return list_rgbd


def load_pairs_from_association_file(file_path, imgs_root_path, valid_timestamps, valid_poses, order_depth_rgb=True,
                                     time_threshold=0.05):
    """
    从关联文件中加载符合条件的 RGB 和深度图路径对，并绑定位姿。
    """
    print(f"Loading pairs from association file: {file_path}")

    list_rgbd = []

    # 注意：这里不能转 set，因为需要通过 index 找到 valid_poses 中对应的位姿
    # 如果 valid_timestamps 很大，可以使用 list(valid_timestamps) 确保它是序列
    v_ts = list(valid_timestamps)

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            if line.startswith("#"): continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            if order_depth_rgb:
                timestamp_raw, depth_path, _, rgb_path = parts[:4]
            else:
                timestamp_raw, rgb_path, _, depth_path = parts[:4]

            timestamp_raw = float(timestamp_raw)

            # 查找匹配项的索引
            # 使用 enumerate 可以在找到匹配时间戳的同时拿到索引 i
            matched_idx = next((i for i, ts in enumerate(v_ts) if abs(ts - timestamp_raw) < time_threshold), None)

            if matched_idx is not None:
                matched_ts = v_ts[matched_idx]
                matched_pose = valid_poses[matched_idx]  # ✨ 核心修改：通过索引获取对应位姿

                diff = abs(matched_ts - timestamp_raw)

                # 构造完整路径
                depth_full_path = os.path.join(imgs_root_path, depth_path)
                rgb_full_path = os.path.join(imgs_root_path, rgb_path)

                # 将位姿也存入字典
                list_rgbd.append({
                    'timestamp': matched_ts,
                    'fpath_depth': depth_full_path,
                    'fpath_rgb': rgb_full_path,
                    'pose': matched_pose  # ✨ 绑定位姿
                })
            # else:
            #     print("未找到Valid时间戳： ", timestamp_raw)

    print(f"✅ 成功加载并对齐图像与位姿: {len(list_rgbd)} 对")
    return list_rgbd



def load_poses_from_file(poses_file, skip_every_n=1):
    """
    从 KeyFrames_for_smp.txt 文件中读取位姿数据。

    每行格式: time x y z qx qy qz qw
    返回:
        poses: List[Tuple[int, List[float]]]，每个元素是 (time, [x, y, z, qx, qy, qz, qw])
    """
    print(f"Loading valid_timestamps from KeyFrames Pose file: {poses_file}")

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
                    time = float(parts[0])
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

    print(f"导入 {len(poses)} 个有效时间戳 ")

    return valid_timestamps, poses