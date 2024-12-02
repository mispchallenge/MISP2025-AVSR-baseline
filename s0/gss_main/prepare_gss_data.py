#!/usr/bin/env python
# -- coding: UTF-8 
import glob
import argparse
from multiprocessing import Pool
import sys
import os, re, codecs, jieba
from zhon.hanzi import punctuation
import numpy as np


def text2lines(textpath, lines_content=None):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
    if lines_content is None:
        with codecs.open(textpath, 'r') as handle:
            lines_content = handle.readlines()
        processed_lines = list(map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content))
        return processed_lines
    else:
        processed_lines = list(map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), lines_content))
        with codecs.open(textpath, 'w') as handle:
            handle.write(''.join(processed_lines))
        return None

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

# class definition
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
                # import pdb;pdb.set_trace()
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
        # import pdb;pdb.set_trace()
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
                # import pdb;pdb.set_trace()
                word_list.extend(jieba.cut(text_segmentation, HMM=False))
                
            # self.content = ' '.join(word_list) #jieba 分词空格
            self.content = ''.join(word_list) #
        return None
    
    def text(self, prefix):
        return '{}_{:0>7d}-{:0>7d} {}'.format(
            prefix, int(1000*round(self.xmin, 3)), int(1000*round(self.xmax, 3)), self.content)


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
            if not xmin > xmax:
                intervals.append(Interval(xmin=xmin, xmax=xmax, content=content))
        tiers.append(Tier(tclass=tclass, name=name, xmin=tier_xmin, xmax=tier_xmax, intervals=intervals))
        # import pdb;pdb.set_trace()
    tg = TextGrid(file_type=file_type, object_class=object_class, xmin=tg_xmin, xmax=tg_xmax, tiers=tiers)
    return tg

# channels.scp <recording-id> <extended-filename>
def prepare_channels_scp(data_root, store_dir, set_type='train'):#set_type train dev eval  sub_dir:far mid near

    all_wav_lines = []
    wav_path_list = glob.glob(os.path.join(data_root, '*wav'))
    wav_path_list = list(set([ path[:-6] for path in wav_path_list])) #  path[:-6] :01.wav path[:-4]:.wav
    print(wav_path_list)
    for wav_path in wav_path_list:
        record_id = os.path.split(wav_path)[-1]
        all_wav_lines.append('{} {}'.format(record_id, wav_path))
    if not os.path.exists('{}/temp'.format(store_dir)):
        os.makedirs('{}/temp'.format(store_dir))
    text2lines(textpath='{}/temp/channels.scp'.format(store_dir), lines_content=all_wav_lines)
    return None

# wav.scp <recording-id> <extended-filename>
def prepare_wav_scp(data_root, store_dir, set_type='train'):#set_type train dev eval  sub_dir:far mid near

    all_wav_lines = []
    # print(f"here,data_root:{os.path.join(data_root,set_type,sub_dir)},store_dir:{store_dir}")
    # import pdb;pdb.set_trace()
    wav_path_list = glob.glob(os.path.join(data_root, '*wav'))
    for wav_path in wav_path_list:
        record_id = os.path.split(wav_path)[-1].split('.')[0]
        all_wav_lines.append('{} {}'.format(record_id, wav_path))
    if not os.path.exists('{}/temp'.format(store_dir)):
        os.makedirs('{}/temp'.format(store_dir))
    text2lines(textpath='{}/temp/wav.scp'.format(store_dir), lines_content=all_wav_lines)
    return None

def prepare_mp4_scp(data_root, store_dir, set_type='train'):
    mp4_path_list = glob.glob(os.path.join(data_root, '*mp4'))
    all_mp4_lines = []
    for mp4_path in mp4_path_list:
        if "v4" not in mp4_path:
            mp4_id = os.path.split(mp4_path)[-1].split('.')[0]
            all_mp4_lines.append('{} {}'.format(mp4_id, mp4_path))
    if not os.path.exists('{}/temp'.format(store_dir)):
        os.makedirs('{}/temp'.format(store_dir))
    text2lines(textpath='{}/temp/mp4.scp'.format(store_dir), lines_content=all_mp4_lines)
    return None

# segments <utterance-id> <recording-id> <segment-begin> <segment-end>
# text <utterance-id> <words>
# utt2spk <utterance-id> <speaker-id>
def prepare_segments_text_utt2spk_worker(transcription_dir, set_type, store_dir, processing_id=None, processing_num=None,pos="far",without_wav=False,timestamp_fileroot=None):
    segments_lines = []
    text_sentence_lines = []
    utt2spk_lines = []
    tier_name = '内容层'
    rejected_text_list = ['<NOISE>', '<DEAF>','<moving  sound>','<其他说话人>','<非会议内容>','<knock>','<sil>','<sound  of  door>','<cough>','<laughter>','<NOISE> ','<NOISE-MUSIC>','<OVERLAP>','<NOISE>*<NOISE>','<NOISE>  ','<NOISE>QU','<telephone  ring>','<英文>']+ ['*'*i for i in range(1, 30)]
    punctuation_list = ['!','！','。','.' ,',','，','?','？','、',' ']
    sound_list = ['呃', '啊', '噢', '嗯', '唉','<NOISE>','**']
    # sound_list = []
    min_duration = 0.12
    # import pdb;pdb.set_trace()
    if not without_wav:
        wav_lines = sorted(text2lines(textpath='{}/temp/wav.scp'.format(store_dir), lines_content=None))
    else:
        wav_lines = sorted(text2lines(textpath='{}/temp/channels.scp'.format(store_dir), lines_content=None))
    for wav_idx in range(len(wav_lines)):
        # import pdb;pdb.set_trace()
        if processing_id is None:
            processing_token = True
        else:
            if wav_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            wav_id, wav_path = wav_lines[wav_idx].split(' ')
            start_time, end_time = None, None

           
            # room, speakers, config, index = wav_id.split('_')[:4]
            room, speakers, config = wav_id.split('_')[:3]
            speaker_list = [speakers[i: i+3] for i in range(1, len(speakers), 3)]
            # if pos=="near":
            #     speaker_list = [wav_id.split('_')[-1]]  # 修改
            for speaker in speaker_list:
                # import pdb;pdb.set_trace();
                # import pdb;pdb.set_trace()
                checkout_tg = read_textgrid_from_file(filepath=os.path.join(transcription_dir, '{}_{}_{}_Near_{}.TextGrid'.format(room, speakers, config, speaker)))
                tg = checkout_tg.cutoff(xstart=start_time, xend=end_time)
                target_tier = False
                for tier in tg.tiers:
                    if tier.name == tier_name:
                        target_tier = tier
                if not target_tier:
                    raise ValueError('no tier: {}'.format(tier_name))
                # import pdb;pdb.set_trace()
                for interval in target_tier.intervals:
                    # if interval.text not in rejected_text_list and interval.xmax - interval.xmin >= min_duration:
                    # if interval.content=='<telephonering>':
                        # import pdb;pdb.set_trace()
                    if interval.content not in rejected_text_list and interval.xmax - interval.xmin >= min_duration:
                        start_stamp = interval.xmin - interval.xmin % 0.04
                        start_stamp = round(start_stamp, 2)
                        end_stamp = interval.xmax + 0.04 - interval.xmax % 0.04 if interval.xmax % 0.04 != 0 else \
                            interval.xmax
                        end_stamp = round(end_stamp, 2)
                        utterance_id = 'S{}_{}_{}_{}_'.format(speaker, room, speakers, config) + \
                                       '{0:06d}'.format(int(round(start_stamp*100, 0))) + '-' + \
                                       '{0:06d}'.format(int(round(end_stamp*100, 0)))
                        # text = interval.text
                        text = interval.content
                        # import pdb;pdb.set_trace()
                        for punctuation in punctuation_list:
                            text = text.replace(punctuation, '')
                        if text not in sound_list:
                            
                            segments_lines.append('{} {} {} {}'.format(utterance_id, wav_id, start_stamp, end_stamp))
                            text_sentence_lines.append('{} {}'.format(utterance_id, text))
                            utt2spk_lines.append('{} S{}'.format(utterance_id, speaker))
    # import pdb;pdb.set_trace()
    return [segments_lines, text_sentence_lines, utt2spk_lines]


def prepare_segments_text_utt2spk_manager(transcription_dir, set_type, store_dir, processing_num=1,pos="far",without_wav=False):
    
    if processing_num > 1:
        pool = Pool(processes=processing_num)
        all_result = []
        for i in range(processing_num):
            
            part_result = pool.apply_async(prepare_segments_text_utt2spk_worker, kwds={
                'transcription_dir': transcription_dir, 'set_type': set_type, 'store_dir': store_dir, 'processing_id': i, 
                'processing_num': processing_num,'without_wav': without_wav})
            all_result.append(part_result)
        pool.close()
        pool.join()
        segments_lines, text_sentence_lines, utt2spk_lines = [], [], []
        for item in all_result:
            part_segments_lines, part_text_sentence_lines, part_utt2spk_lines = item.get()
            segments_lines += part_segments_lines
            text_sentence_lines += part_text_sentence_lines
            utt2spk_lines += part_utt2spk_lines
    else:
        segments_lines, text_sentence_lines, utt2spk_lines = prepare_segments_text_utt2spk_worker(
            transcription_dir=transcription_dir, set_type=set_type, store_dir=store_dir,pos=pos,without_wav=without_wav)
    text2lines(textpath='{}/temp/segments'.format(store_dir), lines_content=segments_lines)
    text2lines(textpath='{}/temp/text_sentence'.format(store_dir), lines_content=text_sentence_lines)
    text2lines(textpath='{}/temp/utt2spk'.format(store_dir), lines_content=utt2spk_lines)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('wav_dir', type=str, default='', help='directory of wav')
    parser.add_argument('mp4_dir', type=str, default='', help='directory of mp4')
    parser.add_argument('transcription_dir', type=str, default='', help='directory of transcription')
    parser.add_argument('set_type', type=str, default='', help='set type')
    parser.add_argument('store_dir', type=str, default='data/train_far', help='set types')
    parser.add_argument('--channel_dir', type=str, default=None, help='directory of original multiple channel waveform')
    parser.add_argument('--without_wav', type=bool, default=False, help='don not prepare wav.scp')
    parser.add_argument('--without_mp4', type=bool, default=True, help='don not prepare mp4.scp')
    parser.add_argument('--without_others', type=bool, default=False, help='do not prepare segments,text_sentence,utt2spk')
    parser.add_argument('-nj', type=int, default=1, help='number of process')


    args = parser.parse_args()
    #pos decides the number of speakeers in a wav, for near field audio there is only one, for far or middel there can be more
    #the arg will change the mode how to generate text in def prepare_segments_text_utt2spk_worker 
    if "near" in args.store_dir:
        pos="near"
    else:
        pos="far"
    #channels.scp
    if args.channel_dir:
        print('Preparing channels.scp in {}'.format(args.store_dir))
        prepare_channels_scp(data_root=args.channel_dir, store_dir=args.store_dir, set_type=args.set_type)
    #wav.scp
    if not args.without_wav:
        print('Preparing wav.scp in {}'.format(args.store_dir))
        prepare_wav_scp(data_root=args.wav_dir, store_dir=args.store_dir, set_type=args.set_type)
    # import pdb;pdb.set_trace()
    #mp4.scp similar to wav.scp
    if not args.without_mp4:
        print('Preparing mp4.scp in {}'.format(args.store_dir))
        prepare_mp4_scp(data_root=args.mp4_dir, store_dir=args.store_dir, set_type=args.set_type)

    if not args.without_others:

        #segments,text_sentence,utt2spk
        print('Preparing segments,text_sentence,utt2spk in {}'.format(args.store_dir))
        
        prepare_segments_text_utt2spk_manager(
            transcription_dir=args.transcription_dir, set_type=args.set_type, store_dir=args.store_dir, processing_num=args.nj,pos=pos,without_wav=args.without_wav)

    