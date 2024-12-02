#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os, re, codecs, jieba
import numpy as np
#from zhon.hanzi import punctuation

def list_str_match(pattern_lst, str_lst):
    pattern_len = len(pattern_lst)
    if pattern_len != len(str_lst):
        raise ValueError('unmatched len of pattern lst {} and str lst {}'.format(pattern_len, len(str_lst)))
    value_lst =[]
    for i in range(pattern_len):
        value_candidate = re.findall(pattern_lst[i], str_lst[i])
        if len(value_candidate) == 1:
            value_lst.append(value_candidate[0])
        else:
            raise ValueError('unmatched pattern {} and str {}'.format(pattern_lst[i], str_lst[i]))
    return value_lst

class Interval(object):
    def __init__(self, xmin=0., xmax=0., content=''):
        self.xmin = xmin
        self.xmax = xmax
        self.content = content

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))
    
    def numpy(self, sr=16000):
        sli_signs = ['<其他说话人>', '<NOISE>']
        point_num = int(round((self.xmax - self.xmin) * sr))
        if self.content in sli_signs:
            return np.zeros(point_num)
        else:
            return np.ones(point_num)
    
    def word_segmentation(self):
        no_word_signs = ['<其他说话人>', '<NOISE>', '<主说话人>', '<非会议内容>'] + ['*'*i for i in range(1, 30)]
        if self.content not in no_word_signs:
            text_list = list(filter(lambda y: y!= '', re.split('|'.join(list(punctuation)), self.content)))
            word_list = []
            for text_segmentation in text_list:
                word_list.extend(jieba.cut(text_segmentation, HMM=False))
            self.content = ' '.join(word_list)
        return None
    
    def text(self, prefix):
        return '{}-{:0>7d}-{:0>7d} {}'.format(
            prefix, int(1000*round(self.xmin, 3)), int(1000*round(self.xmax, 3)), self.content)

class Tier(object):
    def __init__(self, tclass='', name='', xmin=None, xmax=None, intervals=[]):
        self.tclass = tclass
        self.name = name
        self.intervals = intervals
        self.xmin = xmin if xmin is not None else self.intervals[0].xmin
        self.xmax = xmax if xmax is not None else self.intervals[-1].xmax
        
        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))
        
        x = self.xmin
        for i, interval in enumerate(self.intervals):
            if interval.xmin != x:
                raise ValueError('NO.{} interval is not continuous, need {} but got {}'.format(i, x, interval.xmin))
            else:
                x = interval.xmax
        if x!= self.xmax:
            raise ValueError('There is a gap between the last interval and the end of the Tier, from {} to{}'.format(x, self.xmax))

    def cutoff(self, xstart=None, xend=None):
        if xstart is None:
            xstart = self.xmin

        if xend is None:
            xend = self.xmax

        if xend < xstart:
            raise ValueError('xend ({}) < xstart ({})'.format(xend, xstart))

        bias = xstart - self.xmin
        new_xmax = xend - bias
        new_xmin = self.xmin
        new_intervals = []
        for interval in self.intervals:
            if interval.xmax <= xstart or interval.xmin >= xend:
                pass
            elif interval.xmin < xstart:
                new_intervals.append(Interval(xmin=new_xmin, xmax=interval.xmax - bias, content=interval.content))
            elif interval.xmax > xend:
                new_intervals.append(Interval(xmin=interval.xmin - bias, xmax=new_xmax, content=interval.content))
            else:
                new_intervals.append(Interval(xmin=interval.xmin - bias, xmax=interval.xmax - bias, content=interval.content))

        return Tier(tclass=self.tclass, name=self.name, xmin=new_xmin, xmax=new_xmax, intervals=new_intervals)
    
    def numpy(self, sr=16000):
        interval_arrays = []
        for interval in self.intervals:
            interval_arrays.append(interval.numpy(sr=sr))
        return np.concatenate(interval_arrays)
    
    def word_segmentation(self):
        for i in range(len(self.intervals)):
            self.intervals[i].word_segmentation()
        return None
    
    def text(self, prefix):
        no_word_signs = ['<其他说话人>', '<NOISE>', '<主说话人>', '<非会议内容>'] + ['*'*i for i in range(1, 30)]
        text_lines = []
        for interval in self.intervals:
            if interval.content not in no_word_signs:
                text_lines.append(interval.text(prefix=prefix))
        return text_lines

# class definition
class TextGrid(object):
    def __init__(self, file_type='', object_class='', xmin=0., xmax=0., tiers=[]):
        self.file_type = file_type
        self.object_class = object_class
        self.tiers = tiers
        self.xmin = xmin if xmin is not None else self.tiers[0].xmin
        self.xmax = xmax if xmax is not None else self.tiers[0].xmax
    
        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))
        for i, tier in enumerate(self.tiers[1:]):
            if tier.xmin != xmin or tier.xmax != xmax:
                raise ValueError('NO.{} tier is out of sync, should begin at {} but {} and end at {} but {}'.format(i, self.xmin, xmin, self.xmax, xmax))   
    
    def cutoff(self, xstart=None, xend=None):
        if xstart is None:
            xstart = self.xmin

        if xend is None:
            xend = self.xmax

        if xend < xstart:
            raise ValueError('xend ({}) < xstart ({})'.format(xend, xstart))

        new_xmax = xend - xstart + self.xmin
        new_xmin = self.xmin
        new_tiers = []

        for tier in self.tiers:
            new_tiers.append(tier.cutoff(xstart=xstart, xend=xend))
        return TextGrid(file_type=self.file_type, object_class=self.object_class, xmin=new_xmin, xmax=new_xmax,
                        tiers=new_tiers)
    
    def numpy(self, sr=16000):
        for tier in self.tiers:
            if tier.name == '内容层':
                return tier.numpy(sr=16000)
    
    def word_segmentation(self):
        for i in range(len(self.tiers)):
            if self.tiers[i].name == '内容层':
                self.tiers[i].word_segmentation()
        return None
    
    def text(self, prefix, filepath=None, word_segmentation=False):
        used_i = 0
        for i in range(len(self.tiers)):
            if self.tiers[i].name == '内容层':
                used_i = i
                break
        
        used_tiers = self.tiers[used_i]
        if word_segmentation:
            used_tiers.word_segmentation()
        
        text_lines = used_tiers.text(prefix=prefix)
        if filepath is None:
            return text_lines
        else:
            store_dir = os.path.split(filepath)[0]
            if not os.path.exists(store_dir):
                os.makedirs(store_dir, exist_ok=True)
            processed_lines = [*map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), text_lines)]
            with codecs.open(filepath, 'w') as handle:
                handle.write(''.join(processed_lines))
            return None


        
def read_textgrid_from_file(filepath):
    with codecs.open(filepath, 'r', encoding='utf8') as handle:
        lines = list(filter(lambda y: y!= '', map(lambda x: x.strip().replace('"', ''), handle.readlines())))
    
    file_type, object_class, tg_xmin, tg_xmax, tiers_state, tiers_size = list_str_match(
            pattern_lst=[
                'File type = (\w+)', 'Object class = (\w+)', 'xmin = (\d+\.?\d*)', 
                'xmax = (\d+\.?\d*)', 'tiers\? (.+)', 'size = (\d+\.?\d*)'], 
            str_lst=lines[:6])
    tg_xmin, tg_xmax, tiers_size = float(tg_xmin), float(tg_xmax), int(tiers_size)        
    
    tiers = []
    tiers_idxes = []
    for i in range(tiers_size):
        tiers_idxes.append(lines.index('item [{}]:'.format(i + 1)))
    tiers_idxes.append(len(lines))
    
    for i in range(tiers_size):
        tier_lines = lines[tiers_idxes[i]+1: tiers_idxes[i+1]]
        tclass, name, tier_xmin, tier_xmax, intervals_size =  list_str_match(
            pattern_lst=[
                'class = (\w+)', 'name = (\w+)', 'xmin = (\d+\.?\d*)', 
                'xmax = (\d+\.?\d*)', 'intervals: size = (\d+\.?\d*)'], 
            str_lst=tier_lines[:5])
        tier_xmin, tier_xmax, intervals_size = float(tier_xmin), float(tier_xmax), int(intervals_size)
    
        intervals = []
        intervals_idxes = []
        for j in range(intervals_size):
            intervals_idxes.append(tier_lines.index('intervals [{}]:'.format(j + 1)))
        intervals_idxes.append(len(tier_lines))
        
        for j in range(intervals_size):
            xmin, xmax, content =  list_str_match(
                pattern_lst=[
                    'xmin = (\d+\.?\d*)', 'xmax = (\d+\.?\d*)', 'text = (.+)',], 
            str_lst=tier_lines[intervals_idxes[j]+1: intervals_idxes[j+1]])
            xmin, xmax = float(xmin), float(xmax)
            intervals.append(Interval(xmin=xmin, xmax=xmax, content=content))
        tiers.append(Tier(tclass=tclass, name=name, xmin=tier_xmin, xmax=tier_xmax, intervals=intervals))
    tg = TextGrid(file_type=file_type, object_class=object_class, xmin=tg_xmin, xmax=tg_xmax, tiers=tiers)
    return tg


if __name__ == '__main__':
    checkout_tg = read_textgrid_from_file(filepath='/train33/sppro/permanent/hangchen2/pandora/egs/misp2024-mmmt/data/MISP-Meeting/training/A213/A213-F8N/A213-F8N-219.TextGrid')
    #print(checkout_tg.file_type, checkout_tg.object_class, checkout_tg.xmin, checkout_tg.xmax, checkout_tg.tiers)
    cut_tg = checkout_tg.cutoff(xstart=2.328, xend=7348.437)
    cut_tg.text(prefix='219-A213', filepath='/train33/sppro/permanent/hangchen2/pandora/egs/misp2024-mmmt/data/MISP-Meeting/training/A213/A213-F8N/A213-F8N-219.txt', word_segmentation=True)
