#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os, re, codecs, jieba
import numpy as np
import shutil
import argparse
#from zhon.hanzi import punctuation

def textgrid_cutoff(textgrid_file, timestamp_file, output_file):
    with open(textgrid_file, 'r') as f:
        textgrid_lines = f.readlines()

    xmin = float(textgrid_lines[3].split('=')[1].strip())
    xmax = float(textgrid_lines[4].split('=')[1].strip())

    with open(timestamp_file, 'r') as f:
        timestamp = f.readlines()

    xstart = 0.0
    xend = 0.0

    for i in range(len(timestamp[0].split(':'))):
        tmp = float(timestamp[0].split(':')[i])
        xstart = xstart*60 + tmp

    if len(timestamp) > 1:
        for i in range(len(timestamp[1].split(':'))):
            tmp = float(timestamp[1].split(':')[i])
            xend = xend*60 + tmp
    else:
        xend = xmax

    with open(output_file, 'w') as fout:
        xmin = float(textgrid_lines[3].split('=')[1].strip())
        xmax = float(textgrid_lines[4].split('=')[1].strip())

        for i in range(8):
            if 'xmin = ' in textgrid_lines[i]:
                xmin_glo = float(textgrid_lines[i].split('=')[1].strip())
                line = textgrid_lines[i]
            elif 'xmax = ' in textgrid_lines[i]:
                tmp = float(textgrid_lines[i].split('=')[1].strip())
                xmax_glo = round(xend - xstart + xmin_glo, 5)
                line = textgrid_lines[i].replace(str(tmp), str(xmax_glo))
            else:
                line = textgrid_lines[i]
            fout.write(line)
            #print(line)
        #print(xstart, xend, xmin_glo, xmax_glo)
        bias = xstart - xmin_glo
        for i in range(8, len(textgrid_lines), 1):
            if 'xmin = ' in textgrid_lines[i]:
                xmin = float(textgrid_lines[i].split('=')[1].strip())
                if xmin < xstart:
                    line = textgrid_lines[i].replace(str(xmin), str(xmin_glo))
                else:
                    new_xmin = xmin - bias
                    line = textgrid_lines[i].replace(str(xmin), str(new_xmin))
            elif 'xmax = ' in textgrid_lines[i]:
                xmax = float(textgrid_lines[i].split('=')[1].strip())
                if xmax > xend:
                    line = textgrid_lines[i].replace(str(xmax), str(xmax_glo))
                else:
                    new_xmax = xmax - bias
                    if new_xmax<0:
                        new_xmax=0.0
                    line = textgrid_lines[i].replace(str(xmax), str(new_xmax))
            else:
                line = textgrid_lines[i]
            fout.write(line)
            #print(line)

def search_files(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith('.TextGrid') and filename.split('.')[0].split('-')[-1] != 'cutoff':
                textgrid_file = os.path.join(root, filename)
                timestamp_file = os.path.join(root, 'timestamp.txt')
                output_filename = filename.split('.')[0]+'-cutoff.TextGrid'
                output_file = os.path.join(root, output_filename)
                if not os.path.exists(timestamp_file):
                    shutil.copy(textgrid_file, output_file)
                else:
                    textgrid_cutoff(textgrid_file, timestamp_file, output_file)
                print(f'{filename} done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('base_dir', type=str, default='', help='directory of source data')
    args = parser.parse_args()

    search_files(args.base_dir)

