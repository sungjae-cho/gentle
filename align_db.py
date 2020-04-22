import os
import gentle
import numpy as np
import pandas as pd
import codecs
import logging
import time
import datetime
import math
import wave
import contextlib
import librosa
import librosa.display
import pathlib
import matplotlib.pyplot as plt
#import sounddevice as sd
import operator
from shutil import copyfile
from tqdm import tqdm
from pprint import pprint
from ffmpy import FFmpeg
from trimmer import get_vad_ranges

# DOWNLOAD THE DB AND CHANGE THIS PATH
#path='/data2/sungjaecho/data_tts/EmoV-DB/EmoV-DB_sorted'
resources = gentle.Resources()
emov_db_path = '/data4/data/EmoV-DB'
emov_db_16000 = '02_EmoV-DB-sr-16000'
#emov_db_version = '01_EmoV-DB-original'
#emov_db_version = '02_EmoV-DB-sr-22050'
#emov_db_version = '02_EmoV-DB-sr-16000'
emov_db_version = '03_EmoV-DB-sr-22050-trim-vad'
path_alignments = 'alignments/EmoV-DB_sorted'
people_list = ['bea', 'jenie', 'josh', 'sam']
emo_list = ['Amused', 'Angry', 'Disgusted', 'Neutral', 'Sleepy']
data_stat_path = 'data_stat'
emo_csv_name = 'emov_db.csv'
path_emov_db = os.path.join(emov_db_path, emov_db_version)
emo_csv_path = os.path.join(data_stat_path, emov_db_version, emo_csv_name)

def on_progress(p):
    for k,v in p.items():
        logging.debug("%s: %s" % (k, v))


def load_emov_db_postprocessed():
    db_path = os.path.join(emov_db_path, '02_EmoV-DB-sr-22050', 'emov_db_postprocessed.xlsx')
    df = pd.read_excel(db_path)
    return df


def load_emov_db(path_to_EmoV_DB=None, load_csv=False, load_script_from_postprocessed=True):
    if load_csv and os.path.exists(emo_csv_path):
        data = load_csv_db()
        print("DB loaded from {} !".format(emo_csv_path))

        return data

    print('Start to load wavs.')

    script = os.path.join(path_to_EmoV_DB, 'cmuarctic.data')
    lines = codecs.open(script, 'r', 'utf-8').readlines()


    df_pp = load_emov_db_postprocessed()

    # in our database, we use only files beginning with arctic_a. And the number of these sentences correspond.
    # Here we build a dataframe with number and text of each of these lines
    sentences = []
    for line in lines:
        temp = {}
        idx_n_0 = line.find('arctic_a') + len('arctic_a')
        if line.find('arctic_a') != -1:
            #print(line)
            #print(idx_n_0)
            idx_n_end = idx_n_0 + 4
            number = line[idx_n_0:idx_n_end]
            #print(number)
            temp['n'] = number
            idx_text_0 = idx_n_end + 2
            text = line.strip()[idx_text_0:-3]
            temp['text'] = text
            # print(text)
            sentences.append(temp)
    sentences = pd.DataFrame(sentences)

    #print(sentences)
    speakers=next(os.walk(path_to_EmoV_DB))[1] #this list directories (and not files, contrary to osl.listdir() )

    data=[]

    for spk in speakers:
        print("Speaker: {}".format(spk))
        emo_cat = next(os.walk(os.path.join(path_to_EmoV_DB,spk)))[1] #this list directories (and not files, contrary to osl.listdir() )

        for emo in emo_cat:
            print("Emotion: {}".format(emo))
            for file in tqdm(os.listdir(os.path.join(path_to_EmoV_DB, spk, emo))):
                #print(file)
                db_dir = os.path.split(path_to_EmoV_DB)[1]
                fpath_abs = os.path.join(path_to_EmoV_DB, spk, emo, file)
                fpath = os.path.join(spk, emo, file)

                if file[-4:] == '.wav':
                    fnumber = file[-8:-4]
                    #print(fnumber)
                    if fnumber.isdigit():
                        if load_script_from_postprocessed:
                            text = get_script_from_df(df_pp, spk, emo, int(fnumber))
                        else:
                            text = sentences[sentences['n'] == fnumber]['text'].iloc[0]  # result must be a string and not a df with a single element
                        # text_lengths.append(len(text))
                        # texts.append(text)
                        # texts.append(np.array(text, np.int32).tostring())
                        # fpaths.append(fpath_abs)
                        # emo_cats.append(emo)

                        duration = get_wav_duration(fpath_abs)

                        e = {'database': 'EmoV-DB',
                             'db_version': emov_db_version,
                             'id': '{}_{}_{}'.format(spk, emo, fnumber),
                             'speaker': spk,
                             'emotion':emo,
                             'script': text,
                             'cmu_a_id': fnumber,
                             'sentence_path': fpath,
                             'duration': duration}
                        data.append(e)
                        #print(e)

    data = pd.DataFrame.from_records(data)
    data = data.sort_values(by=['speaker', 'emotion', 'cmu_a_id'])
    #data = fix_wrong_scripts(data)

    print("DB loaded from {} !".format(path_emov_db))

    return data


def get_audio_files(dir_path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dir_path):
        for file in f:
            if '.wav' == file[-4:]:
                files.append(os.path.join(r, file))

    return files


def get_script_from_df(df, speaker, emotion, cmu_a_id):
    condition = (df.speaker == speaker) & (df.emotion == emotion) & (df.cmu_a_id == cmu_a_id)
    script = df.loc[condition, 'script'].values[0]

    return script


def resampling_audios(dir_audios, dir_target, sample_rate=22050):
    a_paths = get_audio_files(dir_audios)
    print("Make all wavs in {dir_audios} {sample_rate} samples rate and save it in {dir_target}".format(
        dir_audios=dir_audios,
        sample_rate=sample_rate,
        dir_target=dir_target
    ))
    for a_path in tqdm(a_paths):
        dir_path, a_full_name = os.path.split(a_path)
        dir_speaker_path, dir_emo = os.path.split(dir_path)
        dir_db_path, dir_speaker = os.path.split(dir_speaker_path)
        a_name, _ = os.path.splitext(a_full_name)
        str_option = "-ar {sample_rate} -y -hide_banner -loglevel panic".format(sample_rate=sample_rate)
        new_file_path = '{}.wav'.format(os.path.join(dir_target, dir_speaker, dir_emo, a_name))
        ff = FFmpeg(
            inputs={a_path: None},
            outputs={new_file_path: str_option}
        )
        try:
            ff.run()
        except:
            print(new_file_path)
            pass

def fix_wrong_scripts(data):
    condition = (data.speaker == 'sam') & (data.emotion == 'disgusted') & (data.id == 'disgusted_57-84_0075')
    data.loc[condition, 'script'] = 'The gray eyes faltered; the flush deepened.'

    condition = (data.speaker == 'josh') & (data.emotion == 'amused') & (data.id == 'amused_225_252_0026')
    data.loc[condition, 'script'] = 'I came before my ABCs.'

    return data


def align_db(data):

    except_i_list = list(range(len(data)))

    while True:
        for i in tqdm(except_i_list):
            row = data.iloc[i]
            f = row.sentence_path
            script = row.script
            with gentle.resampled(f) as wavfile:
                aligner = gentle.ForcedAligner(resources, script, nthreads=40)
                #print("Align starts")
                try:
                    result = aligner.transcribe(wavfile, progress_cb=on_progress, logging=logging)
                except:
                    except_i_list.append(i)

                    continue
                #print("Align ends")
            # os.system('python align.py '+f+' words.txt -o test.json')

            output = os.path.join('alignments', '/'.join(f.split('/')[-4:]).split('.')[0] + '.json')
            pathlib.Path('/'.join(output.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

            fh = open(output, 'w')
            #print("{} starts to be written".format(output))
            fh.write(result.to_json(indent=2))
            #print("{} written".format(output))
            if output:
                logging.info("output written to %s" % (output))

            fh.close()
            #print("i={}".format(i))
            #print("f={}".format(row.sentence_path))

        print("except_i_list:", except_i_list)
        if len(except_i_list) == 0:
            break


def align_onefile(data, align_json_path):
    splitted_path = split_path(align_json_path)
    json_file_name = splitted_path[-1]
    id, _ = os.path.splitext(json_file_name)
    emotion = splitted_path[-2]
    speaker = splitted_path[-3]


    row = data[(data.id == id) & (data.speaker == speaker)].iloc[0] # iloc: df to series
    f = row.sentence_path
    script = row.script
    with gentle.resampled(f) as wavfile:
        aligner = gentle.ForcedAligner(resources, script, nthreads=40)
        #print("Align starts")
        try:
            result = aligner.transcribe(wavfile, progress_cb=on_progress, logging=logging)
        except:
            return
            #except_i_list.append(i)
            #print("except_i_list:", except_i_list)
        #print("Align ends")
    # os.system('python align.py '+f+' words.txt -o test.json')

    output = os.path.join('alignments', '/'.join(f.split('/')[-4:]).split('.')[0] + '.json')
    pathlib.Path('/'.join(output.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

    fh = open(output, 'w')
    #print("{} starts to be written".format(output))
    fh.write(result.to_json(indent=2))
    #print("{} written".format(output))
    if output:
        logging.info("output written to %s" % (output))

    fh.close()
    #print("i={}".format(i))
    #print("f={}".format(row.sentence_path))


def align_again():
    '''
    Make alignment json files again that do not have the start attribute.
    '''

    data = load_emov_db(path_emov_db)
    while True:
        if len(data) == 0:
            break

        json_path_list = list()
        for i, row in tqdm(data.iterrows()):
            original_wav_path = row.sentence_path
            o_wav_dir, o_wav_name = os.path.split(original_wav_path)
            #json_path = 'alignments/EmoV-DB_sorted/bea/Amused/amused_1-15_0001.json'
            json_path = os.path.join('alignments', 'EmoV-DB_sorted', row.speaker, row.emotion, '{}.json'.format(row.id))
            try:
                start, end = get_start_end_from_json(json_path)
            except Exception as e:
                json_path_list.append(json_path)
                print(e)
                #print(json_path)
        print('#Errors:', len(json_path_list))
        print(json_path_list)

        for json_path in tqdm(json_path_list):
            try:
                align_onefile(data, json_path)
            except Exception as e:
                print(e)

        if len(json_path_list) == 0:
            break

    print("Finish align_again().")


def split_path(path):
    path = os.path.abspath(path)
    path = os.path.normpath(path).split(os.path.sep)

    return path


def get_start_end_from_json(path):
    '''
    Example:
        json_path = 'alignments/EmoV-DB_sorted/bea/disgusted/disgusted_141-168_0149.json'
        start, end = get_start_end_from_json(json_path)
    '''
    #a=pd.read_json(os.path.join('file://localhost', os.path.abspath(path)))
    #print(path)
    a=pd.read_json(os.path.abspath(path))
    a=pd.read_json(path)
    b=pd.DataFrame.from_records(a.words)

    #print('start:')
    start=b.start[0]
    #print(start)

    #print('end:')
    end=b.end.round(2).tolist()[-1]
    #print(end)

    return start, end

def get_start_end_from_df_emovdb(df_emovdb, speaker, emotion, script_id):
    df = df[(df.speaker == speaker) & (df.emotion == emotion) & (df.cmu_a_id == script_id)]
    fa_trim_start = df.iloc[0].trim_start
    fa_trim_end = df.iloc[0].trim_end

    return fa_trim_start, fa_trim_end

# path='alignments/EmoV-DB/bea/amused/amused_1-15_0001.json'
# start, end=get_start_end_from_json(path)

def play_start_end(path, start, end):
    y,fs=librosa.load(path)
    sd.play(y[int(start*fs):int(end*fs)],fs)


def save_wav_start_end(ori_wav_path, new_wav_path, start=None, end=None):
    y,fs=librosa.load(ori_wav_path)

    if (start is None) or (math.isnan(start)):
        start = 0
    if (end is None) or (math.isnan(end)):
        end = y.shape[-1] - 1
    start_word_index = int(start*fs)
    end_word_index = int(end*fs)

    _, trim_index = librosa.effects.trim(y, top_db=20)

    start_trim_index = trim_index[0]
    end_trim_index = trim_index[1]
    #librosa.output.write_wav(new_wav_path, y[start_word_index:],fs)
    # End time are wrong often.

    max_sec_diff = 0.5
    min_sec_diff = 0.1
    start_time_diff = abs(start_word_index - start_trim_index) / (1.0 * fs)
    end_time_diff = abs(end_word_index - end_trim_index) / (1.0 * fs)
    if (min_sec_diff < start_time_diff) and (start_time_diff < max_sec_diff):
        start_index = max(start_word_index, start_trim_index)
    else:
        start_index = start_trim_index
    if (min_sec_diff < end_time_diff) and (end_time_diff < max_sec_diff):
        end_index = min(end_word_index, end_trim_index)
    else:
        end_index = end_trim_index

    start_index = fs * start
    end_index = fs * end
    librosa.output.write_wav(new_wav_path, y[start_index:end_index],fs)

    original_sec = librosa.get_duration(y)
    trim_sec = (end_index - start_index) / (1.0 * fs)
    diff_sec = original_sec - trim_sec

    return diff_sec

def save_wav_start_end_simple(ori_wav_path, new_wav_path, start=None, end=None):
    y,fs=librosa.load(ori_wav_path)
    start_index = int(fs * start)
    end_index = int(fs * end) + 1
    librosa.output.write_wav(new_wav_path, y[start_index:end_index],fs)


def get_wav_duration(wav_path):
    y, fs = librosa.load(wav_path)
    n_frames = len(y)
    seconds =  float(n_frames) / fs

    return seconds

def play(path):
    y,fs=librosa.load(path)
    sd.play(y,fs)

def make_alignments():
    #i_resume = 207
    print("Load DB")
    data=load_emov_db(path_emov_db)
    print("Align DB starts!")
    print("Length of DB: {}".format(len(data)))
    #i_list = [289, 309, 311, 340, 361, 453, 666, 705, 780, 828, 830, 982, 1161, 1164, 1170, 1195, 1635, 1715, 1736, 1865, 2199, 2406, 2428, 2684, 4350, 4351, 4356, 4377, 4392, 4475, 4553, 4558, 4593, 4659, 4705, 4833, 5110, 5233, 5355, 5359, 5495, 5560, 5685, 5692, 5785, 5922, 5954, 6038, 6120, 6557]
    #i_list = [1195]
    align_db(data)
    print("Done")


def trim_wavs_by_vad():
    data = load_emov_db_postprocessed()
    vad_time_start_len_set = set()
    vad_time_end_len_set = set()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        original_wav_path = os.path.join(
            emov_db_path, emov_db_version, row.sentence_path
        )
        o_wav_dir, o_wav_name = os.path.split(original_wav_path)
        start, end, vad_time_start_list, vad_time_end_list = get_vad_trim_range(
            row.speaker, row.emotion, int(row.cmu_a_id)
        )
        try:
            save_wav_start_end_simple(original_wav_path, original_wav_path, start, end)
        except Exception as e:
            if start is None:
                print("start is None")
            if end is None:
                print("end is None")
            print(e)


def trim_wavs_with_start_end():
    diff_trim_start_list = list()
    diff_trim_end_list = list()
    error_json_path_list = list()
    path_diffsec_dict = dict()
    #data = load_emov_db(path_emov_db, load_csv=True)
    data = load_emov_db_postprocessed()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        original_wav_path = os.path.join(
            emov_db_path, emov_db_version, row.sentence_path
        )
        o_wav_dir, o_wav_name = os.path.split(original_wav_path)
        #json_path = 'alignments/EmoV-DB_sorted/bea/Amused/amused_1-15_0001.json'
        #json_path = os.path.join('alignments', 'EmoV-DB_sorted', row.speaker, row.emotion, '{}.json'.format(row.id))
        #start, end = get_start_end_from_json(json_path)
        if row.emotion in ['amused', 'sleepy']:
            start, end, vad_time_start_list, vad_time_end_list = get_shortest_vad_trim_range(
                row.fa_trim_start, row.fa_trim_end,
                row.speaker, row.emotion, int(row.cmu_a_id)
            )

        else:
            start, end, vad_time_start_list, vad_time_end_list = get_vad_trim_range(
                row.speaker, row.emotion, int(row.cmu_a_id)
            )

        try:
            #print(original_wav_path)
            save_wav_start_end_simple(original_wav_path, original_wav_path, start, end)
        except Exception as e:
            if start is None:
                print("start is None")
            if end is None:
                print("end is None")
            print(e)
            #error_json_path_list.append(json_path)
    #print('#Errors:', len(error_json_path_list))
    #pprint(error_json_path_list)
    #with open('trim_except_json_path_list.txt', 'w') as f:
    #    f.write(str(error_json_path_list))
    #pprint(sorted_path_diffsec_dict[:10])
    #sorted_path_diffsec_dict = sorted(path_diffsec_dict, key=path_diffsec_dict.get, reverse=True)
    #diff_trim_start_list = np.asarray(diff_trim_start_list)
    #diff_trim_end_list = np.asarray(diff_trim_end_list)
    #print("len(diff_trim_start_list)", diff_trim_start_list.shape[0])
    #print("len(diff_trim_end_list)", diff_trim_end_list.shape[0])
    #print("max(diff_trim_start_list)", diff_trim_start_list.max())
    #print("max(diff_trim_end_list)", diff_trim_end_list.min())
    #diff_trim_start_mean = diff_trim_start_list.mean()
    #diff_trim_end_mean = diff_trim_end_list.mean()
    #print("mean(diff_trim_start_list)", diff_trim_start_mean)
    #print("mean(diff_trim_end_list)", diff_trim_end_mean)
    #print("count diff_trim_start_list > mean", np.sum(diff_trim_start_list > diff_trim_start_mean))
    #print("count diff_trim_end_list > mean", np.sum(diff_trim_end_mean > diff_trim_end_mean))



def get_vad_trim_range(speaker, emotion, cmu_a_id,
    frame_duration_ms=30, padding_duration_ms=300):

    wav_path = os.path.join(emov_db_path, emov_db_16000, speaker, emotion,
        "{}_{:04d}.wav".format(emotion, cmu_a_id))
    trim_start, trim_end, vad_time_start_list, vad_time_end_list = get_vad_ranges(wav_path, frame_duration_ms, padding_duration_ms)

    return trim_start, trim_end, vad_time_start_list, vad_time_end_list


def get_shortest_vad_trim_range(fa_trim_start ,fa_trim_end, speaker, emotion,
    cmu_a_id, frame_duration_ms=10, padding_duration_ms=30):

    wav_path = os.path.join(emov_db_path, emov_db_16000, speaker, emotion,
        "{}_{:04d}.wav".format(emotion, cmu_a_id))
    _, _, vad_time_start_list, vad_time_end_list = get_vad_ranges(wav_path, frame_duration_ms, padding_duration_ms)
    vad_time_start_list = sorted(vad_time_start_list)
    vad_time_end_list = sorted(vad_time_end_list, reverse=True)

    trim_start, trim_end = None, None

    for vad_time_start in vad_time_start_list:
        if vad_time_start <= fa_trim_start:
            trim_start = vad_time_start
        else:
            break

    for vad_time_end in vad_time_end_list:
        if vad_time_end >= fa_trim_end:
            trim_end = vad_time_end
        else:
            break

    if trim_start is None:
        for vad_time_start in vad_time_start_list:
            if abs(vad_time_start - fa_trim_start) < 0.2:
                trim_start = vad_time_start
                break

    if trim_end is None:
        for vad_time_end in vad_time_end_list:
            if abs(vad_time_end - vad_time_end) < 0.2:
                trim_end = vad_time_end
                break


    broad_trim_start, broad_trim_end, _, _ = get_vad_trim_range(speaker, emotion, cmu_a_id)

    if abs(fa_trim_start - broad_trim_start) > 0.5:
        trim_start = broad_trim_start
    if abs(fa_trim_end - broad_trim_end) > 0.5:
        trim_end = broad_trim_end

    return trim_start, trim_end, vad_time_start_list, vad_time_end_list


def trim_silence(top_db=15):
    data = load_emov_db(path_emov_db)
    for i, row in tqdm(data.iterrows(), total=len(data)):
        original_wav_path = os.path.join(path_emov_db, row.sentence_path)
        dir_path, full_file_name = os.path.split(original_wav_path)
        file_name, file_ext = os.path.splitext(full_file_name)

        y, fs = librosa.load(original_wav_path)
        yt, trim_index = librosa.effects.trim(y, top_db=top_db)
        os.remove(original_wav_path)
        librosa.output.write_wav(original_wav_path, yt, fs)
        #print("Trimmed {}".format(original_wav_path))

        sr = fs

        plt.figure(figsize=(10, 4))
        librosa.display.waveplot(y, sr=sr)
        plt.title('Monophonic\n{}\n{}'.format(original_wav_path, row.script))
        plt.tight_layout()
        plt.axvline(trim_index[0]/fs, color='red')
        plt.axvline(trim_index[1]/fs, color='red')
        wav_img_path = os.path.join(dir_path, "{}.{}".format(file_name, 'png'))
        plt.savefig(wav_img_path)
        #print("Saved {}".format(wav_img_path))
        #plt.show()
        plt.close()

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                             fmax=sr/2)
        plt.figure(figsize=(10, 4))
        S_dB = librosa.power_to_db(S, ref=np.max)

        librosa.display.specshow(S_dB, x_axis='time',
                                 y_axis='mel', sr=sr,
                                fmax=sr/2)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram\n{}\n{}'.format(original_wav_path, row.script))
        plt.tight_layout()
        plt.axvline(trim_index[0]/fs, color='yellow')
        plt.axvline(trim_index[1]/fs, color='yellow')
        mel_img_path = os.path.join(dir_path, "{}_mel.{}".format(file_name, 'png'))
        plt.savefig(mel_img_path)
        #print("Saved {}".format(mel_img_path))
        #plt.show()
        plt.close()

        txt_file_name = '{}.txt'.format(file_name)
        txt_path = os.path.join(dir_path, txt_file_name)
        with open(txt_path, 'w') as f:
            f.write(row.script)
            #print("Saved {}".format(txt_path))


def align_again_nan_start_end():

    data = load_emov_db(path_emov_db)
    while True:
        nan_start_json_path_list = list()
        nan_end_json_path_list = list()
        except_json_path_list = list()
        for i, row in tqdm(data.iterrows(), total=len(data)):
            original_wav_path = row.sentence_path
            o_wav_dir, o_wav_name = os.path.split(original_wav_path)
            json_path = os.path.join('alignments', 'EmoV-DB_sorted', row.speaker, row.emotion, '{}.json'.format(row.id))

            try:
                start, end = get_start_end_from_json(json_path)

                if math.isnan(start):
                    nan_start_json_path_list.append(json_path)
                if math.isnan(end):
                    nan_end_json_path_list.append(json_path)
                if math.isnan(start) or math.isnan(end):
                    align_onefile(data, json_path)
            except Exception as e:
                print(e)
                except_json_path_list.append(json_path)
                align_onefile(data, json_path)

        print('#nan_start:', len(nan_start_json_path_list))
        print('#nan_end:', len(nan_end_json_path_list))
        print('#except:', len(except_json_path_list))
        with open('nan_start_json_path_list.txt', 'w') as f:
            f.write(str(nan_start_json_path_list))
        with open('nan_end_json_path_list.txt', 'w') as f:
            f.write(str(nan_end_json_path_list))
        with open('except_json_path_list.txt', 'w') as f:
            f.write(str(except_json_path_list))
        if len(nan_start_json_path_list) + len(nan_start_json_path_list) == 0:
            break

    print('Finish align_again_nan_start_end.')
    print(except_json_path_list)


def check_wrong_alignments(save_results=True):
    '''
    3 cases of wrong alignments
      1. Get nan at the start argument from get_start_end_from_json.
      2. Get nan at the end argument from get_start_end_from_json.
      3. In the alignment json, neither start nor end attributes exist.
    '''
    data = load_emov_db(path_emov_db)
    nan_start_json_path_list = list()
    nan_end_json_path_list = list()
    except_json_path_list = list()

    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        original_wav_path = row.sentence_path
        o_wav_dir, o_wav_name = os.path.split(original_wav_path)
        json_path = os.path.join('alignments', 'EmoV-DB_sorted', row.speaker, row.emotion, '{}.json'.format(row.id))

        try:
            start, end = get_start_end_from_json(json_path)
            if math.isnan(start):
                nan_start_json_path_list.append(json_path)
            if math.isnan(end):
                nan_end_json_path_list.append(json_path)
        except:
            except_json_path_list.append(json_path)

    print('#nan_start:', len(nan_start_json_path_list))
    print('#nan_end:', len(nan_end_json_path_list))
    print('#except:', len(except_json_path_list))
    if save_results:
        with open('nan_start_json_path_list.txt', 'w') as f:
            f.write(str(nan_start_json_path_list))
        with open('nan_end_json_path_list.txt', 'w') as f:
            f.write(str(nan_end_json_path_list))
        with open('except_json_path_list.txt', 'w') as f:
            f.write(str(except_json_path_list))

def regularizing_file_names():
    data = load_emov_db(path_emov_db)
    id_emo_set = set()
    emotion_set = set()
    for i, row in data.iterrows():
        dir_path_wav, filename_wav = os.path.split(row.sentence_path)
        emo_dir_dir_wav, emo_dir = os.path.split(dir_path_wav)

        filename_wav = replace_file_name(filename_wav.lower())
        splitted_full_filename = filename_wav.split('_')
        emo, txt_num_ext = splitted_full_filename[0], splitted_full_filename[-1]
        filename_wav = "{}_{}".format(emo, txt_num_ext)
        emo_dir = emo_dir.lower()

        old_path = os.path.join(path_emov_db, row.sentence_path)
        new_path = os.path.join(path_emov_db, emo_dir_dir_wav, emo_dir, filename_wav)
        print('From:', old_path)
        print('To  :', new_path)

        new_dir = os.path.join(path_emov_db, emo_dir_dir_wav, emo_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        os.rename(
            old_path,
            new_path
        )

        filename_json = '{}.json'.format(row.id)
        path_json = os.path.join('alignments', 'EmoV-DB_sorted', row.speaker, row.emotion, filename_json)
        emo_dir_dir_json = os.path.join('alignments', 'EmoV-DB_sorted', row.speaker)

        filename_json = replace_file_name(filename_json.lower())
        splitted_full_filename = filename_json.split('_')
        emo, txt_num_ext = splitted_full_filename[0], splitted_full_filename[-1]
        filename_json = "{}_{}".format(emo, txt_num_ext)

        old_path = path_json
        new_path = os.path.join(emo_dir_dir_json, emo_dir, filename_json)

        print('From:', old_path)
        print('To  :', new_path)
        print()

        new_dir = os.path.join(emo_dir_dir_json, emo_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        #os.rename(
        #    old_path,
        #    new_path
        #)
    return

def replace_file_name(string):
    string = string.replace('sleepiness_', 'sleepy')
    string = string.replace('disgust_', 'disgusted')
    string = string.replace('anger_', 'angry_')
    return string

def copy_wavs_for_manual_alignment():
    json_path_set = {
        'alignments/EmoV-DB_sorted/sam/neutral/neutral_281-308_0308.json',
        'alignments/EmoV-DB_sorted/josh/sleepy/sleepy_281-308_0288.json',
        'alignments/EmoV-DB_sorted/josh/neutral/neutral_1-28_0021.json',
        'alignments/EmoV-DB_sorted/josh/neutral/neutral_169-196_0182.json',
        'alignments/EmoV-DB_sorted/josh/neutral/neutral_29-56_0032.json',
        'alignments/EmoV-DB_sorted/sam/neutral/neutral_309-336_0330.json',
        'alignments/EmoV-DB_sorted/bea/sleepy/sleepy_393-420_0407.json',
        'alignments/EmoV-DB_sorted/josh/neutral/neutral_57-84_0075.json',
        'alignments/EmoV-DB_sorted/bea/angry/angry_197-224_0024.json',
        'alignments/EmoV-DB_sorted/josh/amused/amused_197-224_0023.json',
        'alignments/EmoV-DB_sorted/sam/disgusted/disgusted_253-280_0264.json',
        'alignments/EmoV-DB_sorted/sam/disgusted/disgusted_336-364_0345.json',
        'alignments/EmoV-DB_sorted/sam/sleepy/sleepy_281-308_0387.json',
        'alignments/EmoV-DB_sorted/josh/neutral/neutral_57-84_0074.json',
        'alignments/EmoV-DB_sorted/josh/neutral/neutral_57-84_0067.json'
    }
    wav_path_set = set()

    new_dir = '/data2/sungjaecho/data_tts/EmoV-DB/wavs_for_manual_alignment'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    for json_path in tqdm(json_path_set):
        splitted_path = split_path(json_path)
        speaker = splitted_path[-3]
        emotion = splitted_path[-2]
        json_name = os.path.splitext(splitted_path[-1])[0]

        wav_name = '{}.wav'.format(json_name)
        wav_path = os.path.join(path_emov_db, speaker, emotion, wav_name)
        new_wav_path = os.path.join(new_dir, speaker, emotion, wav_name)

        speaker_dir = os.path.join(new_dir, speaker)
        if not os.path.exists(speaker_dir):
            os.mkdir(speaker_dir)

        emotion_dir = os.path.join(new_dir, speaker, emotion)
        if not os.path.exists(emotion_dir):
            os.mkdir(emotion_dir)

        copyfile(wav_path, new_wav_path)

    print("Finished copying wavs for manual alignment.")


def save_db_to_csv(with_trim_time=True):
    data = load_emov_db(path_emov_db)

    if with_trim_time:
        print("Add trim_start and trim_end columns storing time seconds where to trim.")
        trim_start_list = list()
        trim_end_list = list()

        for i, row in tqdm(data.iterrows(), total=len(data)):
            original_wav_path = row.sentence_path
            o_wav_dir, o_wav_name = os.path.split(original_wav_path)
            #json_path = 'alignments/EmoV-DB_sorted/bea/Amused/amused_1-15_0001.json'
            json_path = os.path.join('alignments', 'EmoV-DB_sorted',
                row.speaker, row.emotion, '{}.json'.format(row.id))
            start, end = get_start_end_from_json(json_path)
            trim_start_list.append(start)
            trim_end_list.append(end)

        data['fa_trim_start'] = trim_start_list
        data['fa_trim_end'] = trim_end_list

        # Re-order columns
        cols = [
            'database', 'db_version', 'id', 'speaker', 'emotion', 'cmu_a_id', 'script',
            'fa_trim_start', 'fa_trim_end', 'sentence_path', 'duration'
        ]
    else:
        # Re-order columns
        cols = [
            'database', 'db_version', 'id', 'speaker', 'emotion', 'cmu_a_id', 'script',
            'sentence_path', 'duration'
        ]
    data = data[cols]

    # 1
    saved_csv_path = os.path.join(path_emov_db, emo_csv_name)
    data.to_csv(saved_csv_path, index=False)
    print("The generated CSV file saved in {}.".format(saved_csv_path))
    # 2
    dir_data_stat_db = os.path.join(data_stat_path, emov_db_version)
    if not os.path.exists(dir_data_stat_db):
        os.mkdir(dir_data_stat_db)
    saved_csv_path = os.path.join(dir_data_stat_db, emo_csv_name)
    data.to_csv(saved_csv_path, index=False)
    print("The generated CSV file saved in {}.".format(saved_csv_path))


def load_csv_db():
    df = pd.read_csv(emo_csv_path)

    return df


def get_hms_time(seconds):
    hms_time = datetime.timedelta(seconds=seconds)

    return hms_time

def get_db_stat():
    '''
    Get the statistics of DB.
    '''
    if not os.path.exists(data_stat_path):
        os.mkdir(data_stat_path)

    df = load_emov_db(load_csv=True)

    print('Get the sum of all durations.')
    df_all = df['duration'].agg(['size','min', 'max', 'mean', 'std', 'sum']).to_frame().reset_index()
    df_all = df_all.rename(columns={'index':'stat'})
    df_all = df_all.append({'stat': 'sum_hms', 'duration':get_hms_time(df_all[df_all.stat == 'sum']['duration'].values[0])}, ignore_index=True)

    print('Grouped by speakers.')
    df_by_spk= df.groupby(['speaker'])['duration'].agg(['size','min', 'max', 'mean', 'std', 'sum'])
    df_by_spk['sum_hms'] = df_by_spk['sum'].apply(get_hms_time)

    print('Grouped by emotions.')
    df_by_emo = df.groupby(['emotion'])['duration'].agg(['size','min', 'max', 'mean', 'std', 'sum'])
    df_by_emo['sum_hms'] = df_by_emo['sum'].apply(get_hms_time)

    print('Grouped by speakers and emotions.')
    df_by_spk_emo = df.groupby(['speaker', 'emotion'])['duration'].agg(['size','min', 'max', 'mean', 'std', 'sum'])
    df_by_spk_emo['sum_hms'] = df_by_spk_emo['sum'].apply(get_hms_time)

    save_path = os.path.join(data_stat_path, 'df_all.csv')
    df_all.to_csv(save_path)
    print('Saved in {} !'.format(save_path))

    save_path = os.path.join(data_stat_path, 'df_by_spk.csv')
    df_by_spk.to_csv(save_path)
    print('Saved in {} !'.format(save_path))

    save_path = os.path.join(data_stat_path, 'df_by_emo.csv')
    df_by_emo.to_csv(save_path)
    print('Saved in {} !'.format(save_path))

    save_path = os.path.join(data_stat_path, 'df_by_spk_emo.csv')
    df_by_spk_emo.to_csv(save_path)
    print('Saved in {} !'.format(save_path))


if __name__ == "__main__":
    #regularizing_file_names()
    #save_db_to_csv(with_trim_time=False)
    #trim_silence()
    '''resampling_audios(
        '/data2/sungjaecho/data_tts/EmoV-DB/02_EmoV-DB-sr-22050',
        '/data2/sungjaecho/data_tts/EmoV-DB/02_EmoV-DB-sr-16000',
        16000
    )'''
    #make_alignments()
    #align_again_nan_start_end()
    #check_wrong_alignments()
    #align_again()
    #trim_wavs_by_vad()
    #trim_wavs_with_start_end()
    save_db_to_csv(with_trim_time=False)
    #get_db_stat()

    pass
