# 脚本功能解释
    + pose_RVIZ.py ：   ROS 可视化节点，用于从 .npz 文件中读取相机的绝对位姿数据，将相机坐标系下的位姿转换到世界坐标系后，通过 RViz 的 Marker 显示相机运动过程：每一帧用红色球体表示相机在世界坐标系中的位置，用绿色箭头表示相机的前向朝向，并按时间顺序逐帧发布，从而直观地展示相机的运动轨迹和姿态变化。
    + Projection3D_RVIZ.py ： 读取"NPZ文件"，语义分割结果与深度图结合生成3D语义点云，并通过 ROS 显示。
    + mit_segment_NPZ_ICL_3D_project.py : 
        + 读取"SLAM数据集"，对 RGB‑D 图像进行语义分割，将结果映射为 27 类室内语义并上色保存，
        + 结合相机位姿把深度数据投影到三维空间，通过 ROS 在 RViz 中可视化语义点云。
        + 保存原图、深度图、语义标签和位姿信息到 .npz 文件
    + ProjectionGround_from_NPZ.py : 读取"NPZ文件"，将语义分割结果与深度图结合生成3D语义点云，并投影到地面。将地面投影添加进    NPZ文件

# 保存SLAM-RGBD数据到NPZ文件中
    + 实现了语义分割
    + 实现了地面投影
    + 实现了ego和geo之间的变换，但似乎有问题，
    + 将ego保存到NPZ文件中
        + 

# 使用顺序：
    + mit_segment_NPZ_ICL_3D_project.py 
    + ProjectionGround_from_NPZ.py