import os
import textgrid

def check_textgrid(src_path):
    files_with_hyphen = []
    for filename in os.listdir(src_path):
        if filename.endswith(".TextGrid"):
            file_path = os.path.join(src_path, filename)
            try:
                tg = textgrid.TextGrid.fromFile(file_path)
                for tier in tg.tiers:
                    for interval in tier.intervals:
                        for interval in tier.intervals:
                            if '-' in interval.mark and filename not in files_with_hyphen:
                                files_with_hyphen.append(filename)
            except Exception as e:
                print(f"Error {filename}: {e}")
    return files_with_hyphen

src_path = '/train33/sppro/permanent/hangchen2/pandora/egs/misp2024-mmmt/data/MISP-Meeting/training_transcription_rename'
files_with_hyphen = check_textgrid(src_path)
print(files_with_hyphen)