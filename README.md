# 安装方法
  <!-- + conda create -n seg_torch_env python=3.9 -y
  + conda activate seg_torch_env
  + conda install numpy=1.26.4 -y
  + conda install matplotlib
  + conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y -->

  conda create -n semseg python=3.9
  conda activate semseg_2
  pip install numpy==1.26.4 numpy-quaternion==2023.0.4 # 使用以上版本
  pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
  pip install -e .
  conda install matplotlib
  pip install numpy-quaternion
  pip install rospkg

  
# 多机器人分布
  + 修改两个电脑的bashrc
      export ROS_MASTER_URI=http://192.168.4.56:11311
      export ROS_IP=192.168.4.56
      export ROS_HOSTNAME=192.168.4.56
  + 修改主机的/etc/hosts
      127.0.0.1	localhost
      127.0.1.1	robotlab-MECHREV-local
      192.168.4.56  robotlab-MECHREV
  + 

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
## 生成语义分割Image，检查语义分割是否有问题
  + 使用方法：bash demo_test.sh  #
  + 修改demo_test.sh文件
    + TEST_IMG 保存图片的位置
  + 修改mit_segment_NPZ.py文件：
    + TEST_IMG：    保存rgb原图的地方  
    + TEST.result： 保存语义分割结果的地方
    + imgs_sampled = imgs[::20]  每隔多少张图片识别一次
    
## 根据RGB-D和SLAM-Pose生成NPZ文件（all_data.npz）
  + 运行方法：bash demo_assoication.sh
  + 文件： mit_segment_NPZ_ICL_3D_project.py 
  + 数据集处理：
    + SLAM轨迹： KeyFrames_for_smp.txt
  + 修改demo_assoication.sh文件：
    + 图片的位置： IMG_ROOT=/media/robotlab/WD_BLACK/RuiHaiJiaYuan/mylivingroom
    + RGBD关联文件： ASSOCIATION_FILE_NAME=associations_smp.txt
    + 保存结果的位置： ./save_results/temp
  + 修改python文件：
    + 修改 skip_steps 和 type_name
    + 是否开启ROS可视化 flag_3D_rviz = False
  + 用途： 
    + 保存 彩色图、深度图、语义分割图、pose(xyz+四元数)，位置在cfg.TEST.result, 名字为all_data.npz
    + ROS:发布无语义的点云，话题为visualization_marker
  + DEBUG:
    + 可以通过注释掉default_marker_id += 1，实现单帧深度图在
  + 注意：有效时间戳的数量应该与匹配到的


## 查看 上一步生成的NPZ文件
  + 使用方法： python ground_projection/visualize_NPZ_without_step_ego_grid_27_all.py
  
## 生成语义地面投影图，并加入NPZ文件（virtual_robot_outputs.npz）
  + 使用方法： python ground_projection/ProjectionGround_from_NPZ.py
  + 修改python文件：
      + 要处理的NPZ文件位置： npz_file_path = "/home/robotlab/work/semantic-segmentation-pytorch/save_results/temp/all_data.npz"
      + 点云的高度筛选:  h = local3D_step[:, 2] # 只使用两米内的点云
  + Debug方法：
    + 在transform_global_to_ego_single函数中，对于机器人地面位姿的[x,y,theta]三个参数，只开启其中一个，用于查看它是否正确。
  + 用途：
    + 生成 语义地面投影图、pose(x、y、theta)
      + 注意：当前在EGO地图的裁切过程（transform_global_to_ego_single）中，只使用了平移，未使用旋转。
  + 结果： ./save_results/temp/virtual_robot_outputs.np

## 查看 上一步生成的NPZ文件
  + 使用方法： python ground_projection/visualize_NPZ_all.py
  + 修改python文件：
      + NPZ文件位置： npz_file_path = "/home/robotlab/work/semantic-segmentation-pytorch/save_results/temp/virtual_robot_outputs.npz"
      + 快速浏览的步长： step=10 #表示从 0 开始，每次增加 10

## 转入RSMP-Net
  + 使用方法： 
    + 将virtual_robot_outputs.npz复制到/home/robotlab/dataset/semantic/semantic_datasets/**NEW_SCENE_NAME**/test/only_one路径下
    + cd /home/robotlab/semantic-map-prediction
    + python main.py --name slam_binzhou_wjl_2026_2_22.1 --ensemble_dir  path-model/smp   --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs     --stored_episodes_dir /home/robotlab/dataset/semantic/semantic_datasets/**NEW_SCENE_NAME**/    --save_nav_images
    + python main.py --name slam_binzhou_wjl_2026_2_22.1 --ensemble_dir  path-model/smp   --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs     --stored_episodes_dir /home/robotlab/dataset/semantic/semantic_datasets/data_binzhou_wjl/    --save_nav_images
    +  python main.py --name slam_binzhou_wjl_2026_2_22.1   --ensemble_dir  path-model/   --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs    --sem_map_test --stored_episodes_dir /home/robotlab/dataset/semantic/semantic_datasets/data_v6/   --ensemble_size 4
  + 结果图片的保存位置：