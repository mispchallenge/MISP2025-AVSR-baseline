#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import codecs, re
import numpy as np
import sys

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
    def __init__(self, xmin=0., xmax=0., text=''):
        self.xmin = xmin
        self.xmax = xmax
        self.text = text

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))


class Tier(object):
    def __init__(self, tier_class='', name='', xmin=0., xmax=0., intervals=[]):
        self.tier_class = tier_class
        self.name = name
        self.intervals = intervals
        self.xmin = xmin if xmin is not None else self.intervals[0].xmin
        self.xmax = xmax if xmax is not None else self.intervals[-1].xmax
        
        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))

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
                new_intervals.append(Interval(xmin=new_xmin, xmax=interval.xmax - bias, text=interval.text))
            elif interval.xmax > xend:
                new_intervals.append(Interval(xmin=interval.xmin - bias, xmax=new_xmax, text=interval.text))
            else:
                new_intervals.append(Interval(xmin=interval.xmin - bias, xmax=interval.xmax - bias, text=interval.text))

        return Tier(tier_class=self.tier_class, name=self.name, xmin=new_xmin, xmax=new_xmax, intervals=new_intervals)
    def text(self, prefix):
        no_word_signs = ['<其他说话人>', '<NOISE>', '<主说话人>', '<非会议内容>'] + ['*'*i for i in range(1, 30)]
        text_lines = []
        for interval in self.intervals:
            if interval.content not in no_word_signs:
                text_lines.append(interval.text(prefix=prefix))
        return text_lines

# class definition
class TextGrid(object):
    def __init__(self, file_type='', object_class='', xmin=0., xmax=0., tiers_status='', tiers=[]):
        self.file_type = file_type
        self.object_class = object_class
        self.tiers = tiers
        self.xmin = xmin if xmin is not None else self.tiers[0].xmin
        self.xmax = xmax if xmax is not None else self.tiers[0].xmax
        self.tiers_status = tiers_status

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))

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
                        tiers_status=self.tiers_status, tiers=new_tiers)



# io
def read_textgrid_from_file_before(filepath):
    with codecs.open(filepath, 'r', encoding='utf8') as handle:
        lines = handle.readlines()
    if lines[-1] == '\r\n':
        lines = lines[:-1]

    assert 'File type' in lines[0], 'error line 0, {}'.format(lines[0])
    file_type = lines[0].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert 'Object class' in lines[1], 'error line 1, {}'.format(lines[1])
    object_class = lines[1].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert lines[2] == '\r\n', 'error line 2, {}'.format(lines[2])

    assert 'xmin' in lines[3], 'error line 3, {}'.format(lines[3])
    xmin = float(lines[3].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'xmax' in lines[4], 'error line 4, {}'.format(lines[4])
    xmax = float(lines[4].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'tiers?' in lines[5], 'error line 5, {}'.format(lines[5])
    tiers_status = lines[5].split('?')[1].replace(' ', '').replace('\r', '').replace('\n', '')

    assert 'size' in lines[6], 'error line 6, {}'.format(lines[6])
    size = int(lines[6].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert lines[7] == 'item []:\r\n', 'error line 7, {}'.format(lines[7])

    tier_start = []
    for item_idx in range(size):
        tier_start.append(lines.index(' ' * 4 + 'item [{}]:\r\n'.format(item_idx + 1)))

    tier_end = [*tier_start[1:], len(lines)]

    tiers = []
    for tier_idx in range(size):
        tiers.append(read_tier_from_lines(tier_lines=lines[tier_start[tier_idx] + 1: tier_end[tier_idx]]))

    return TextGrid(file_type=file_type, object_class=object_class, xmin=xmin, xmax=xmax, tiers_status=tiers_status,
                    tiers=tiers)

def read_textgrid_from_file(filepath):
    with codecs.open(filepath, 'r', encoding='utf8') as handle:
        print(filepath)
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
        tier_class, name, tier_xmin, tier_xmax, intervals_size =  list_str_match(
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
            try:
                xmin, xmax, text =  list_str_match(
                    pattern_lst=[
                        'xmin = (\d+\.?\d*)', 'xmax = (\d+\.?\d*)', 'text = (.+)',], 
                str_lst=tier_lines[intervals_idxes[j]+1: intervals_idxes[j+1]])
                xmin, xmax = float(xmin), float(xmax)
                intervals.append(Interval(xmin=xmin, xmax=xmax, text=text))
            except Exception as e:
                print(f"Exception: {e}")
        tiers.append(Tier(tier_class=tier_class, name=name, xmin=tier_xmin, xmax=tier_xmax, intervals=intervals))
    tg = TextGrid(file_type=file_type, object_class=object_class, xmin=tg_xmin, xmax=tg_xmax, tiers=tiers)
    return tg


def write_textgrid_to_file(filepath, textgrid):
    lines = [
        'File type = "{}"\r\n'.format(textgrid.file_type),
        'Object class = "{}"\r\n'.format(textgrid.object_class),
        '\r\n',
        'xmin = {}\r\n'.format(textgrid.xmin),
        'xmax = {}\r\n'.format(textgrid.xmax),
        'tiers? {}\r\n'.format(textgrid.tiers_status),
        'size =  {}\r\n'.format(len(textgrid.tiers)),
        'item []:\r\n'
    ]
    for tier_idx, tier in enumerate(textgrid.tiers):
        lines.append(' ' * 4 + 'item [{}]:\r\n'.format(tier_idx + 1))
        lines.extend(write_tier_to_lines(tier=tier))

    lines.append('\r\n')
    
    with codecs.open(filepath, 'w', encoding='utf8') as handle:
        handle.write(''.join(lines))
    return None


def read_tier_from_lines(tier_lines):
    assert 'class' in tier_lines[0], 'error line 0, {}'.format(tier_lines[0])
    tier_class = tier_lines[0].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert 'name' in tier_lines[1], 'error line 1, {}'.format(tier_lines[1])
    name = tier_lines[1].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert 'xmin' in tier_lines[2], 'error line 2, {}'.format(tier_lines[2])
    xmin = float(tier_lines[2].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'xmax' in tier_lines[3], 'error line 3, {}'.format(tier_lines[3])
    xmax = float(tier_lines[3].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'intervals: size' in tier_lines[4], 'error line 4, {}'.format(tier_lines[4])
    intervals_num = int(tier_lines[4].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))
    
    if tier_lines[-1] == '\n':
        tier_lines = tier_lines[:-1]

    assert len(tier_lines[5:]) == intervals_num * 5 or len(tier_lines[5:]) == intervals_num * 4, 'error lines'

    intervals = []
    for intervals_idx in range(intervals_num):
        if len(tier_lines[5:]) == intervals_num * 5:
            assert tier_lines[5 + 5 * intervals_idx + 0] == ' ' * 8 + 'intervals [{}]:\r\n'.format(intervals_idx + 1)
            assert tier_lines[5 + 5 * intervals_idx + 1] == ' ' * 8 + 'intervals [{}]:\r\n'.format(intervals_idx + 1)
            intervals.append(read_interval_from_lines(
                interval_lines=tier_lines[7 + 5 * intervals_idx: 10 + 5 * intervals_idx]))
        else:
            assert tier_lines[5 + 4 * intervals_idx + 0] == ' ' * 8 + 'intervals [{}]:\r\n'.format(intervals_idx + 1)
            intervals.append(read_interval_from_lines(
                interval_lines=tier_lines[6 + 4 * intervals_idx:  + 9 + 4 * intervals_idx]))
    return Tier(tier_class=tier_class, name=name, xmin=xmin, xmax=xmax, intervals=intervals)


def write_tier_to_lines(tier):
    tier_lines = [
        ' ' * 8 + 'class = "{}"\r\n'.format(tier.tier_class),
        ' ' * 8 + 'name = "{}"\r\n'.format(tier.name),
        ' ' * 8 + 'xmin = {}\r\n'.format(tier.xmin),
        ' ' * 8 + 'xmax = {}\r\n'.format(tier.xmax),
        ' ' * 8 + 'intervals: size = {}\r\n'.format(len(tier.intervals)),
    ]

    for interval_idx, interval in enumerate(tier.intervals):
        tier_lines.append(' ' * 8 + 'intervals [{}]:\r\n'.format(interval_idx + 1))
        tier_lines.append(' ' * 8 + 'intervals [{}]:\r\n'.format(interval_idx + 1))
        tier_lines.extend(write_interval_to_lines(interval=interval))
    return tier_lines


def read_interval_from_lines(interval_lines):
    assert len(interval_lines) == 3, 'error lines'

    assert 'xmin' in interval_lines[0], 'error line 0, {}'.format(interval_lines[0])
    xmin = float(interval_lines[0].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'xmax' in interval_lines[1], 'error line 1, {}'.format(interval_lines[1])
    xmax = float(interval_lines[1].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'text' in interval_lines[2], 'error line 2, {}'.format(interval_lines[2])
    text = interval_lines[2].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    return Interval(xmin=xmin, xmax=xmax, text=text)


def write_interval_to_lines(interval):
    interval_lines = [
        ' ' * 12 + 'xmin = {}\r\n'.format(interval.xmin),
        ' ' * 12 + 'xmax = {}\r\n'.format(interval.xmax),
        ' ' * 12 + 'text = "{}"\r\n'.format(interval.text),
    ]
    return interval_lines


if __name__ == '__main__':
    checkout_tg = read_textgrid_from_file(filepath='D:\\Code\\python_project\\Embedding_Aware_Speech_Enhancement_edition_3\\Textgrid_C0001\\1.TextGrid')
    cut_tg = checkout_tg.cutoff(xstart=220)
    write_textgrid_to_file(filepath='D:\\Code\\python_project\\Embedding_Aware_Speech_Enhancement_edition_3\\Textgrid_C0001\\1_cut_220.TextGrid', textgrid=cut_tg)
