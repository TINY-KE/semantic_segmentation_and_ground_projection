color_mapping_27 = {
    0:  (255, 255, 255),   # 白色 white                       空类别 / 无类别 (void)
    1:  (128, 128, 0),     # 橄榄色 olive                     椅子 (chair)
    2:  (0, 0, 255),       # 蓝色 blue                        门 (door)
    3:  (255, 0, 0),       # 红色 red                         桌子 (table)
    4:  (255, 0, 255),     # 洋红色 magenta                   靠垫 / 坐垫 (cushion)
    5:  (0, 255, 255),     # 青色 cyan                        沙发 (sofa)
    6:  (255, 165, 0),     # 橙色 orange                      床 (bed)
    7:  (255, 255, 0),     # 黄色 yellow                      植物 (plant)
    8:  (128, 128, 128),   # 灰色 gray                        洗手池 / 水槽 (sink)
    9:  (128, 0, 0),       # 栗色 maroon                      马桶 (toilet)
    10: (255, 20, 147),    # 深粉红 deep pink                 电视 / 显示器 (tv_monitor)
    11: (0, 128, 0),       # 深绿色 dark green               淋浴器 (shower)
    12: (128, 0, 128),     # 紫色 purple                      浴缸 (bathtub)
    13: (0, 128, 128),     # 水鸭色 teal                      操作台 / 工作台 (counter)
    14: (0, 0, 128),       # 藏青色 navy                     家电 (appliances)
    15: (210, 105, 30),    # 巧克力色 chocolate              建筑结构 (structure)
    16: (188, 143, 143),   # 褐玫瑰色 rosy brown             其他 / 杂项 (other)
    17: (0, 255, 0),       # 绿色 green                      空闲空间 / 可行走区域 (free-space)   ****
    18: (255, 215, 0),     # 金色 gold                       图片 / 挂画 (picture)
    19: (0, 0, 0),         # 黑色 black                      橱柜 / 柜子 (cabinet)
    20: (192, 192, 192),   # 银色 silver                     抽屉柜 (chest_of_drawers)
    21: (138, 43, 226),    # 蓝紫色 blue violet              凳子 (stool)
    22: (255, 127, 80),    # 珊瑚色 coral                    毛巾 (towel)
    23: (238, 130, 238),   # 紫罗兰色 violet                 壁炉 (fireplace)
    24: (245, 245, 220),   # 米色 / 浅卡其 beige            健身器材 (gym_equipment)
    25: (139, 69, 19),     # 马鞍棕 saddle brown            座位（综合类）(seating)
    26: (64, 224, 208)     # 绿松石色 turquoise              衣物 (clothes)
}


# old_idx → new_idx 映射表（只包含有效项）
DEFAULT_NEW_IDX = 0  # 默认类别（void）
# old_to_new_idx = {
#     20: 1,    # 椅子
#     31: 1,
#     76: 1,
#     111: 1,
#     15: 2,    # 门
#     34: 3,    # 桌子
#     16: 3,
#     65: 3,
#     40: 4,    # 靠垫cushion
#     58: 4,
#     24: 5,    # 沙发
#     8: 6,     # 床
#     # 18: 7,    # 植物
#     # 48: 8,    # 水槽
#     90: 10,   # 电视
#     # 146: 11,  # shower 淋浴
#     38: 12,   # 浴缸 bathtub
#     46: 13,   # 柜台 counter
#     71: 13,
#     100: 13,
#     1: 15,    # 墙 structure
#     # 4: 17,    # 地板 free-space
#     # 14: 17,
#     # 23: 18,   # 画
#     11: 19,   # 橱柜 cabinet
#     # 50: 23,   # 壁炉 fireplace
#     # 82: 22    # 毛巾 towel
# }

# 注意old idx需要减一
old_to_new_idx_origin = {
    20: 1,    # 椅子
    31: 1,
    76: 1,
    111: 1,
    15: 2,    # 门
    34: 3,    # 桌子
    16: 3,
    65: 3,
    40: 4,    # 靠垫cushion
    58: 4,
    24: 5,    # 沙发
    8: 6,     # 床
    # 18: 7,    # 植物
    48: 8,    # 水槽
    90: 10,   # 电视
    146: 11,  # shower 淋浴
    38: 12,   # 浴缸 bathtub
    46: 13,   # 柜台 counter
    71: 13,
    100: 13,
    1: 15,    # 墙 structure
    4: 17,    # 地板 free-space
    14: 17,
    29: 17,
    95: 17,
    23: 18,   # 画
    11: 19,   # 橱柜 cabinet
    50: 23,   # 壁炉 fireplace
    82: 22    # 毛巾 towel
}

old_to_new_idx_binzhou_wjl = {
    20: 1,    # 椅子
    31: 1,
    76: 1,
    111: 1,
    15: 2,    # 门
    34: 3,    # 桌子
    16: 3,
    65: 3,
    40: 4,    # 靠垫cushion
    58: 4,
    24: 5,    # 沙发
    8: 6,     # 床
    18: 7,    # 植物
    48: 8,    # 水槽
    66: 9,      # 马桶
    90: 10,   # 电视
    146: 11,  # shower 淋浴
    38: 12,   # 浴缸 bathtub
    46: 13,   # 柜台 counter
    71: 13,
    100: 13,
    1: 15,    # 墙 structure
    19: 15,    # 墙 structure
    4: 17,    # 地板 free-space
    14: 17,
    29: 17,
    95: 17,
    23: 10,   # 画 伪装为 电视
    11: 19,   # 橱柜 cabinet
    50: 23,   # 壁炉 fireplace
    82: 22    # 毛巾 towel
}



# 参考  【金山文档 | WPS云文档】 color_coding_semantic_segmentation_classes  # https://www.kdocs.cn/l/ctMNlLgiSfOu
# 注意old idx需要减一
old_to_new_idx = {
    20: 1,    # 椅子
    31: 1,
    76: 1,
    111: 1,
    15: 2,    # 门
    34: 3,    # 桌子
    16: 3,
    65: 3,
    40: 4,    # 靠垫cushion
    58: 4,
    24: 5,    # 沙发
    8: 6,     # 床
    18: 7,    # 植物
    48: 8,    # 水槽
    90: 10,   # 电视
    146: 11,  # shower 淋浴
    38: 12,   # 浴缸 bathtub
    46: 13,   # 柜台 counter
    71: 13,
    100: 13,
    1: 15,    # 墙 structure
    19: 15,    # 墙 structure
    4: 17,    # 地板 free-space
    14: 17,
    29: 17,
    95: 17,
    23: 18,   # 画
    11: 19,   # 橱柜 cabinet
    50: 23,   # 壁炉 fireplace
    82: 22    # 毛巾 towel
}

def get_Id_Converter(scene_type="binzhou_wjl"):
    if scene_type == "binzhou_wjl":
        return old_to_new_idx_binzhou_wjl
    elif scene_type == "ICL":
        return old_to_new_idx
    else:
        return old_to_new_idx_origin
