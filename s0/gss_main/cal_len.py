# import pandas as pd

# # 读取文件
# input_file = '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/data/train_far/segments'
# df = pd.read_csv(input_file, sep=' ', header=None)

# # 计算差值并添加新列
# df['difference'] =(df.iloc[:, -1] - df.iloc[:, -2]).round(2)

# # 保存结果
# output_file = '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/data/train_far/output_segment.txt'
# df.to_csv(output_file, sep=' ', header=False, index=False)

# print(f'Output saved to {output_file}')

# import pandas as pd

# # 读取文件
# input_file = '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/data/train_far/segments'
# df = pd.read_csv(input_file, sep=' ', header=None)

# # 计算差值
# difference =(df.iloc[:, -1] - df.iloc[:, -2]).round(2)

# # 保存结果
# output_file = '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/data/train_far/output_segment_1.txt'
# difference.to_csv(output_file, sep=' ', header=False, index=False)

# print(f'Output saved to {output_file}')


import pandas as pd
import re

# 读取文件
input_file = '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/data/train_far/text'
df = pd.read_csv(input_file, sep=' ', header=None, quoting=3)  # quoting=3 to handle quotes properly

# 定义正则表达式匹配非中文字符
non_chinese_regex = re.compile(r'[^\u4e00-\u9fff]+')

# 筛选包含非中文字符的行
non_chinese_rows = df[df.iloc[:, 1].apply(lambda x: bool(non_chinese_regex.search(x)))]

# 保存结果
output_file = '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/data/train_far/non_chinese_text.txt'
non_chinese_rows.to_csv(output_file, sep=' ', header=False, index=False)

print(f'Output saved to {output_file}')