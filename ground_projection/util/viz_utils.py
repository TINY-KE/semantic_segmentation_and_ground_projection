
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
import torch
from PIL import Image
# from habitat_sim.utils.common import d3_40_colors_rgb
import ground_projection.util.map_utils as map_utils

'''
MP3D original semantic labels and reduced set correspondence
# Original set from here: https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
0 void 0
1 wall 15 structure
2 floor 17 free-space
3 chair 1
4 door 2
5 table 3
6 picture 18
7 cabinet 19
8 cushion 4
9 window 15 structure
10 sofa 5
11 bed 6
12 curtain 16 other
13 chest_of_drawers 20
14 plant 7
15 sink 8
16 stairs 17 free-space
17 ceiling 17 free-space
18 toilet 9
19 stool 21
20 towel 22
21 mirror 16 other
22 tv_monitor 10
23 shower 11
24 column 15 structure
25 bathtub 12
26 counter 13
27 fireplace 23
28 lighting 16 other
29 beam 16 other
30 railing 16 other
31 shelving 16 other
32 blinds 16 other
33 gym_equipment 24
34 seating 25
35 board_panel 16 other
36 furniture 16 other
37 appliances 14
38 clothes 26
39 objects 16 other
40 misc 16 other
'''
# 27 categories which include the 21 object categories in the habitat challenge
label_conversion_40_27 = {-1:0, 0:0, 1:15, 2:17, 3:1, 4:2, 5:3, 6:18, 7:19, 8:4, 9:15, 10:5, 11:6, 12:16, 13:20, 14:7, 15:8, 16:17, 17:17,
                    18:9, 19:21, 20:22, 21:16, 22:10, 23:11, 24:15, 25:12, 26:13, 27:23, 28:16, 29:16, 30:16, 31:16, 32:16,
                    33:24, 34:25, 35:16, 36:16, 37:14, 38:26, 39:16, 40:16}
color_mapping_27 = {
    0:(255,255,255), # white
    1:(128,128,0), # olive (dark yellow)
    2:(0,0,255), # blue
    3:(255,0,0), # red
    4:(255,0,255), # magenta
    5:(0,255,255), # cyan
    6:(255,165,0), # orange
    7:(255,255,0), # yellow
    8:(128,128,128), # gray
    9:(128,0,0), # maroon
    10:(255,20,147), # pink 
    11:(0,128,0), # dark green
    12:(128,0,128), # purple
    13:(0,128,128), # teal
    14:(0,0,128), # navy (dark blue)
    15:(210,105,30), # chocolate
    16:(188,143,143), # rosy brown
    17:(0,255,0), # green
    18:(255,215,0), # gold
    19:(0,0,0), # black
    20:(192,192,192), # silver
    21:(138,43,226), # blue violet
    22:(255,127,80), # coral
    23:(238,130,238), # violet
    24:(245,245,220), # beige
    25:(139,69,19), # saddle brown
    26:(64,224,208) # turquoise
}

# three label classification (0:void, 1:occupied, 2:free)
label_conversion_40_3 = {-1:0, 0:0, 1:1, 2:2, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:2, 17:2,
                    18:1, 19:1, 20:1, 21:1, 22:1, 23:1, 24:1, 25:1, 26:1, 27:1, 28:1, 29:1, 30:1, 31:1, 32:1,
                    33:1, 34:1, 35:1, 36:1, 37:1, 38:1, 39:1, 40:1}
color_mapping_3 = {
    0:(255,255,255), # white
    1:(0,0,255), # blue
    2:(0,255,0), # green
}



def write_img(img, savepath, name):
    # img: T x 3 x dim x dim, assumed normalized
    for i in range(img.shape[0]):
        vis_img = img[i,:,:,:].cpu().numpy()
        vis_img = np.transpose(vis_img, (1,2,0))
        im_path = savepath + str(i) + "_" + name + ".png"
        cv2.imwrite(im_path, vis_img[:,:,::-1]*255.0)



def colorize_grid(grid, color_mapping=27): # to pass into tensorboardX video
    # Input: grid -- B x T x C x grid_dim x grid_dim, where C=1,T=1 when gt and C=41,T>=1 for other
    # Output: grid_img -- B x T x 3 x grid_dim x grid_dim
    grid = grid.detach().cpu().numpy()
    grid_img = np.zeros((grid.shape[0], grid.shape[1], grid.shape[3], grid.shape[4], 3),  dtype=np.uint8)
    if grid.shape[2] > 1:
        # For cells where prob distribution is all zeroes (or uniform), argmax returns arbitrary number (can be true for the accumulated maps)
        grid_prob_max = np.amax(grid, axis=2)
        inds = np.asarray(grid_prob_max<=0.05).nonzero() # if no label has prob higher than k then assume unobserved
        grid[inds[0], inds[1], 0, inds[2], inds[3]] = 1 # assign label 0 (void) to be the dominant label
        grid = np.argmax(grid, axis=2) # B x T x grid_dim x grid_dim
    else:
        grid = grid.squeeze(2)

    if color_mapping==27:
        color_mapping = color_mapping_27
    else:
        color_mapping = color_mapping_3
    for label in color_mapping.keys():
        grid_img[ grid==label ] = color_mapping[label]
    
    return torch.tensor(grid_img.transpose(0, 1, 4, 2, 3), dtype=torch.uint8)



def write_tensor_imgSegm(img, savepath, name, t=None):
    # pred: T x C x dim x dim
    if img.shape[1] > 1:
        img = torch.argmax(img.cpu(), dim=1, keepdim=True) # T x 1 x cH x cW
    img_labels = img.squeeze(1)

    for i in range(img_labels.shape[0]):
        img0 = img_labels[i,:,:]

        vis_img = np.zeros((img0.shape[0], img0.shape[1], 3), dtype=np.uint8)
        for label in color_mapping_27.keys():
            vis_img[ img0==label ] = color_mapping_27[label]
        
        if t is None:
            im_path = savepath + str(i) + "_" + name + ".png"
        else:
            im_path = savepath + name + "_" + str(t) + "_" + str(i) + ".png"
        cv2.imwrite(im_path, vis_img[:,:,::-1])


# def display_sample(rgb_obs, depth_obs, sseg_img=None, savepath=None):
#     # sseg_img is semantic observation from Matterport habitat
#     depth_obs = depth_obs / np.amax(depth_obs) # normalize for visualization
#     rgb_img = Image.fromarray(rgb_obs, mode="RGB")
#     depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")
#
#     if sseg_img is not None:
#         semantic_img = Image.new("P", (sseg_img.shape[1], sseg_img.shape[0]))
#         semantic_img.putpalette(d3_40_colors_rgb.flatten())
#         semantic_img.putdata((sseg_img.flatten() % 40).astype(np.uint8))
#         semantic_img = semantic_img.convert("RGBA")
#
#         arr = [rgb_img, depth_img, semantic_img]
#         n=3
#     else:
#         arr = [rgb_img, depth_img]
#         n=2
#
#     plt.figure(figsize=(12 ,8))
#     for i, data in enumerate(arr):
#         ax = plt.subplot(1, n, i+1)
#         ax.axis('off')
#         plt.imshow(data)
#     if savepath is None:
#         plt.show()
#     else:
#         plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=100)
#     plt.close()
#


def save_visual_steps(test_ds, sg, sem_lbl, abs_pose, ltg, pose_coords, agent_height, save_img_dir_, t):
    
    target_pred = sg.sem_grid[:,sem_lbl,:,:]
    target_pred = target_pred.permute(1,2,0).cpu().numpy()*255.0
    
    target_uncertainty = sg.per_class_uncertainty_map[:,sem_lbl,:,:].permute(1,2,0).cpu().numpy()
    target_uncertainty /= np.amax(target_uncertainty)
    target_uncertainty = target_uncertainty*255.0
    
    color_sem_grid = colorize_grid(sg.sem_grid.unsqueeze(1))
    im = color_sem_grid[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    pose_ = np.asarray(abs_pose).reshape(1,3)
    gt_grid_crops_objects = map_utils.get_gt_crops(pose_, test_ds.pcloud, test_ds.label_seq_objects, agent_height, 
                                            test_ds.grid_dim, test_ds.crop_size, test_ds.cell_size)
    color_gt_crop = colorize_grid(gt_grid_crops_objects.unsqueeze(0))
    im_gt_crop = color_gt_crop[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    # crop viz inputs to 128 x 128
    area_size = 100 # area around the agent to be evaluated
    area_start = int( (im.shape[0] / 2) - (area_size / 2) )
    area_end = int( (im.shape[0] / 2) + (area_size / 2) )
    im = im[area_start:area_end, area_start:area_end,:]
    target_uncertainty = target_uncertainty[area_start:area_end, area_start:area_end,:]
    target_pred = target_pred[area_start:area_end, area_start:area_end,:]

    # translate coords
    ltg[0,0,0] -= area_start
    ltg[0,0,1] -= area_start
    pose_coords[0,0,0] -= area_start
    pose_coords[0,0,1] -= area_start

    arr = [ im,
            target_pred, 
            target_uncertainty
            ]
    n=len(arr)
    plt.figure(figsize=(20 ,15))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        plt.imshow(data)
        if i==0:
            plt.scatter(ltg[0,0,0], ltg[0,0,1], color="magenta", s=50)
            plt.scatter(pose_coords[0,0,0], pose_coords[0,0,1], color="blue", s=50)
    plt.savefig(save_img_dir_+str(t)+'.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


def save_map_pred_steps(spatial_in, spatial_pred, objects_pred, ego_img_segm, save_img_dir_, t):

    color_spatial_in = colorize_grid(spatial_in.unsqueeze(0), color_mapping=3)
    im_spatial_in = color_spatial_in[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    color_spatial_pred = colorize_grid(spatial_pred, color_mapping=3)
    im_spatial_pred = color_spatial_pred[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    color_objects_pred = colorize_grid(objects_pred, color_mapping=27)
    im_objects_pred = color_objects_pred[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    color_ego_img_segm = colorize_grid(ego_img_segm, color_mapping=27)
    im_ego_img_segm = color_ego_img_segm[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    arr = [ im_spatial_in, 
            im_spatial_pred,
            im_objects_pred,
            im_ego_img_segm
            ]
    n=len(arr)
    plt.figure(figsize=(20 ,15))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, n, i+1)
        ax.axis('off')
        plt.imshow(data)
    plt.savefig(save_img_dir_+"map_step_"+str(t)+'.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()



# zhjd
def add_border(img, color=(255, 0, 0), thickness=5):
    # 如果是 (C,H,W)，转为 (H,W,C)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.copy()

    h, w, c = img.shape
    img[:thickness, :, :] = color  # top
    img[-thickness:, :, :] = color  # bottom
    img[:, :thickness, :] = color  # left
    img[:, -thickness:, :] = color  # right
    return img

def to_5d(t):
    t = torch.tensor(t)
    while t.ndim < 5:
        t = t.unsqueeze(0)  # 在最前面添加一个新维度。例如原来是 (64, 64) → 变成 (1, 64, 64)
    return t


# === 用 colorize_grid 上色 ===
def color_and_extract(grid, color_mapping):
    colorized = colorize_grid(to_5d(grid), color_mapping=color_mapping)
    # 输出可能是 (3,H,W) 或 (1,3,H,W) 或 (1,1,3,H,W)
    colorized = torch.tensor(colorized)
    colorized = colorized[0, 0]
    # 现在 colorized 应为 (3,H,W)
    colorized.permute(1, 2, 0) # 转为 (H,W,3)
    colorized_border = add_border(colorized, color=(10, 10, 10), thickness=1)
    return colorized_border

def show_image_color_and_extract(tensor_or_array, title="image", color_mapping=27):
    if isinstance(tensor_or_array, torch.Tensor):
        img = tensor_or_array.detach().cpu().numpy()
    else:
        img = np.array(tensor_or_array)
    img = color_and_extract(img, color_mapping=color_mapping)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_image(tensor_or_array, title="image"):
    if isinstance(tensor_or_array, torch.Tensor):
        img = tensor_or_array.detach().cpu().numpy()
    else:
        img = np.array(tensor_or_array)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_image_sseg_2d_label(tensor_or_array, title="image"):

    if isinstance(tensor_or_array, torch.Tensor):
        img = tensor_or_array.detach().cpu().numpy()
    else:
        img = np.array(tensor_or_array)

    # 2️⃣ 确保是二维标签图 [H,W]
    # assert img.ndim == 2, f" [zhjd-debug] Expected 2D label map, got shape {img.shape}"
    if img.ndim == 5:
        img = img[0, fix_extract, 0]  # 默认显示的是第二维（time）的第 0 帧。
    elif img.ndim == 4:
        img = img[0, fix_extract]
    elif img.ndim == 4:
        img = img[fix_extract]

    H, W = img.shape
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

    # 3️⃣ 将每个 label 转成 RGB
    for lbl, color in color_mapping_27.items():
        mask = img == lbl
        rgb_img[mask] = color

    # 4️⃣ 显示结果
    plt.imshow(rgb_img)
    plt.title(title)
    plt.axis('off')
    plt.show()


# zhjd
def colorEncode(label_map):
    """
    将单通道标签图转换为彩色图像（RGB）。

    参数:
        label_map: numpy.ndarray, shape (H, W)，每个像素是类别 ID
        color_mapping: dict[int, tuple[int, int, int]]，类别 ID → RGB 颜色

    返回:
        RGB 图像: numpy.ndarray, shape (H, W, 3)，dtype=uint8
    """
    color_mapping = color_mapping_27

    # 保证输入是 numpy，并 squeeze 掉多余维度
    if isinstance(label_map, torch.Tensor):
        label_map = label_map.detach().cpu().numpy()

    label_map = np.squeeze(label_map)  # 去掉多余维度，例如 (1, H, W) → (H, W)

    if label_map.ndim != 2:
        raise ValueError(f"[colorEncode] 输入的 label_map 必须是 2D，但实际是 {label_map.shape}")

    h, w = label_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, color in color_mapping.items():
        color_img[label_map == label_id] = color

    return color_img

def save_only_Global_forROS(global_maps_objects, savepath, name):
    """
    以原分辨率保存全局地图（1像素=1栅格）
    参数:
        global_maps_objects: [1, 1, 27, H, W]
        savepath: 保存路径
        name: 文件名前缀
    """
    # 确保保存路径存在
    os.makedirs(savepath, exist_ok=True)

    # 1. 维度提取并转为 Numpy
    # global_maps 形状: [B, T, 27, H, W]
    global_maps = global_maps_objects.detach().cpu().numpy()
    print(global_maps.shape)  # (1, 1, 27, 1000, 1000)
    B, T, C, cH, cW = global_maps.shape

    for t in range(T):
        # 2. 提取当前帧 [27, H, W]
        current_grid = global_maps[0, t]

        # 3. 获取彩色图像
        # 假设 color_and_extract 返回的是 [H, W, 3] 的 RGB 图像 (uint8 或 0-1 float)
        img_rgb = color_and_extract(current_grid, 27)

        # 4. 格式转换逻辑
        # 如果 color_and_extract 返回的是 0-1 的 float，需要转为 0-255 uint8
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)

        # 5. 颜色空间转换 (RGB -> BGR)
        # 因为 OpenCV 使用 BGR 格式保存
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 6. 直接保存
        # cv2.imwrite 会按照 img_bgr 的矩阵维度 (H, W) 创建图像文件
        # 这确保了 120x600 的 Tensor 保存出来就是 120x600 像素的图片
        save_file = os.path.join(savepath, f"{name}.png")
        cv2.imwrite(save_file, img_bgr)

    print(f"✅ 已按原分辨率({cW}x{cH})保存全局地图至: {save_file}")




def load_Global_fromROS(image_path, color_mapping=color_mapping_27):
    """
    读取本地的 RGB 图像，并还原为 [1, 1, 27, H, W] 的语义地图 Tensor。
    
    参数:
        image_path: 本地 png 图像路径
        color_mapping: 颜色映射字典，默认为 color_mapping_27
        
    返回:
        global_maps_objects: 形状为 [1, 1, 27, H, W] 的 FloatTensor
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图像文件: {image_path}")

    # 1. 使用 OpenCV 读取图像 (默认 BGR 格式)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图像文件，请检查文件是否损坏: {image_path}")
        
    # 2. 转换为 RGB 格式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape

    # 3. 创建 2D 标签图 (初始化为 0，即 void 类别)
    label_map = np.zeros((H, W), dtype=np.int64)

    # 4. 颜色逆向匹配：遍历字典，将对应的 RGB 像素替换为类别 ID
    for label_id, color in color_mapping.items():
        # 寻找图像中与当前 color 完全一致的像素
        # color 是一个 tuple, np.array(color) 转换为数组以便进行广播比较
        mask = np.all(img_rgb == np.array(color), axis=-1)
        label_map[mask] = label_id

    # 注意：在保存图像时，你在 color_and_extract 内部调用了 add_border 添加了 (10, 10, 10) 的边框。
    # 因为 (10, 10, 10) 不在 color_mapping_27 中，所以它会自动回退为我们在步骤 3 初始化的 0 (void 类别)。

    # 5. 将 2D 标签图转为 27 通道的 One-Hot 张量 [27, H, W]
    # 我们创建一个由 0.0 组成的浮点数组
    num_classes = 27
    grid = np.zeros((num_classes, H, W), dtype=np.float32)

    for c in range(num_classes):
        # 如果当前像素属于类别 c，对应通道设为 1.0
        grid[c, :, :] = (label_map == c).astype(np.float32)

    # 6. 扩充维度到 [1, 1, 27, H, W]
    # np.expand_dims 可以在指定轴前增加维度
    grid_5d = np.expand_dims(grid, axis=(0, 1))

    # 7. 转为 PyTorch Tensor
    global_maps_objects = torch.from_numpy(grid_5d)

    print(f"✅ 已成功从图片恢复全局语义地图，形状: {global_maps_objects.shape}")
    
    return global_maps_objects
