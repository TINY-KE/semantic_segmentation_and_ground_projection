import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import matplotlib.font_manager as fm


# 方法1：自动检测可用的中文字体
def setup_chinese_font():
    # 尝试多种中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS',
                     'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']

    # 检查系统可用的字体
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    print("可用的字体:", [f for f in chinese_fonts if f in available_fonts])

    # 设置字体
    for font in chinese_fonts:
        if font in available_fonts:
            rcParams['font.sans-serif'] = [font] + rcParams['font.sans-serif']
            rcParams['axes.unicode_minus'] = False
            print(f"使用字体: {font}")
            break
    else:
        print("未找到中文字体，将使用默认字体")


# 设置中文字体
setup_chinese_font()

# 颜色映射
color_mapping_27 = {
    0: (255, 255, 255),  # 白色 white
    1: (128, 128, 0),  # 橄榄色 olive
    2: (0, 0, 255),  # 蓝色 blue
    3: (255, 0, 0),  # 红色 red
    4: (255, 0, 255),  # 洋红色 magenta
    5: (0, 255, 255),  # 青色 cyan
    6: (255, 165, 0),  # 橙色 orange
    7: (255, 255, 0),  # 黄色 yellow
    8: (128, 128, 128),  # 灰色 gray
    9: (128, 0, 0),  # 栗色 maroon
    10: (255, 20, 147),  # 深粉色 pink
    11: (0, 128, 0),  # 深绿色 dark green
    12: (128, 0, 128),  # 紫色 purple
    13: (0, 128, 128),  # 水鸭色 teal
    14: (0, 0, 128),  # 藏青色 navy
    15: (210, 105, 30),  # 巧克力色 chocolate
    16: (188, 143, 143),  # 褐玫瑰色 rosy brown
    17: (0, 255, 0),  # 绿色 green
    18: (255, 215, 0),  # 金色 gold
    19: (0, 0, 0),  # 黑色 black
    20: (192, 192, 192),  # 银色 silver
    21: (138, 43, 226),  # 蓝紫色 blue violet
    22: (255, 127, 80),  # 珊瑚色 coral
    23: (238, 130, 238),  # 紫罗兰色 violet
    24: (245, 245, 220),  # 米色 beige
    25: (139, 69, 19),  # 马鞍棕 saddle brown
    26: (64, 224, 208)  # 绿松石色 turquoise
}

# 对应物体名称（中文）
color_mapping_27_name = {
    0: "void",
    1: "椅子",
    2: "门",
    3: "桌子",
    4: "靠垫",
    5: "沙发",
    6: "床",
    7: "",
    8: "",
    9: "",
    10: "电视",
    11: "",
    12: "浴缸",
    13: "柜台",
    14: "",
    15: "墙",
    16: "",
    17: "",
    18: "",
    19: "橱柜",
    20: "",
    21: "",
    22: "",
    23: "",
    24: "",
    25: "",
    26: ""
}

# 绘图设置
cols = 9
rows = 3
block_size = 1

fig, ax = plt.subplots(figsize=(cols * 1.6, rows * 1.8))
ax.set_xlim(0, cols)
ax.set_ylim(0, rows + 0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.set_title("Color Mapping with Labels (0~26)", fontsize=16)

# 绘制颜色块和编号、名称
for idx in range(27):
    color = color_mapping_27[idx]
    name = color_mapping_27_name.get(idx, "")
    rgb_norm = tuple(c / 255 for c in color)
    col = idx % cols
    row = rows - 1 - (idx // cols)

    # 色块
    rect = patches.Rectangle((col, row + 0.5), block_size, block_size, facecolor=rgb_norm)
    ax.add_patch(rect)

    # 编号（居中）
    ax.text(col + 0.5, row + 1.0, str(idx),
            color='black' if sum(color) > 382 else 'white',
            ha='center', va='center', fontsize=32, weight='bold')

    # 中文名称（在下方）
    if name:
        ax.text(col + 0.5, row + 1.2, name,
                ha='center', va='top', fontsize=50, color='black', rotation=0)

plt.tight_layout()
plt.show()