# import scipy.io
# import pandas as pd
#
# # 加载MAT文件
# data = scipy.io.loadmat('/home/robotlab/work/semantic-segmentation-pytorch/data/color150.mat')
#
# # 提取数据并保存为Excel
# df = pd.DataFrame(data['your_variable_name'])
# df.to_excel('/home/robotlab/work/semantic-segmentation-pytorch/data/color150.xlsx', index=False)

import scipy.io
import pandas as pd
import os

def mat_to_excel(mat_path, excel_output_path):
    # 加载 .mat 文件（v7.2 及以下）
    data = scipy.io.loadmat(mat_path)

    # 删除 MATLAB 系统变量 (__header__, __version__, __globals__)
    data = {k: v for k, v in data.items() if not k.startswith('__')}

    # 创建一个 Pandas Excel writer
    with pd.ExcelWriter(excel_output_path) as writer:
        for var_name, var_value in data.items():
            # 尝试将变量转为 DataFrame
            try:
                # 一维数组 → 转为列向量
                if isinstance(var_value, (list, tuple)) or (hasattr(var_value, 'ndim') and var_value.ndim == 1):
                    var_value = pd.DataFrame(var_value)
                # 多维数组 → 转为 DataFrame（只支持 2D）
                elif hasattr(var_value, 'ndim') and var_value.ndim == 2:
                    var_value = pd.DataFrame(var_value)
                else:
                    # 转换失败，写入文本形式
                    var_value = pd.DataFrame([str(var_value)])

                # 写入 Excel 的一个工作表，表名为变量名
                var_value.to_excel(writer, sheet_name=var_name[:31])  # Excel 限制 sheet 名最多 31 字符
                print(f"✅ 写入变量 {var_name} 到 Excel")
            except Exception as e:
                print(f"⚠️ 无法写入变量 {var_name}：{e}")

    print(f"\n🎉 成功输出 Excel 文件：{excel_output_path}")


# 示例用法
if __name__ == '__main__':
    mat_file = "/home/robotlab/work/semantic-segmentation-pytorch/data/color150.mat"            # 替换为你的 .mat 文件路径
    excel_file = "output.xlsx"          # 输出的 Excel 文件路径
    mat_to_excel(mat_file, excel_file)