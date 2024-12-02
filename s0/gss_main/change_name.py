# 修改音频文件的名字使得其可以保证名字和视频文件对齐
import os
rootdir ='/disk4/hblan/graduate/gss_main/eval_wave/far/wpe/gss_new/enhanced'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith(".wav"):
            # 提取出原文件名中的部分作为新文件名
            name_parts = file.split("_")
            start_time = int(name_parts[-2][4:])
            end_time = int(name_parts[-1][0:-4])
            if start_time % 4 != 0:
                res = start_time % 4
                start_time = start_time + 4 - res
            if end_time % 4 != 0:
                res = end_time % 4
                end_time = end_time + 4 - res
            start_time = str(start_time).zfill(6)
            end_time = str(end_time).zfill(6)
            new_name = f"{name_parts[0]}_{name_parts[1]}_{name_parts[2]}_{name_parts[3]}_Far-{start_time}_{end_time}.wav"

            # 构造旧文件名和新文件名的绝对路径
            old_path = os.path.join(subdir, file)
            new_path = os.path.join(subdir, new_name)

            # 重命名文件
            os.rename(old_path, new_path)