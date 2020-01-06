import os
import gentle
import pandas as pd
import codecs
import logging
import time
import datetime
import math
import wave
import contextlib
from shutil import copyfile
from tqdm import tqdm
from pprint import pprint

# DOWNLOAD THE DB AND CHANGE THIS PATH
#path='/data2/sungjaecho/data_tts/EmoV-DB/EmoV-DB_sorted'
resources = gentle.Resources()
path_emov_db='/data2/sungjaecho/data_tts/EmoV-DB/EmoV-DB_sorted'
path_alignments = 'alignments/EmoV-DB_sorted'
people_list = ['bea', 'jenie', 'josh', 'sam']
emo_list = ['Amused', 'Angry', 'Disgusted', 'Neutral', 'Sleepy']
emo_csv_path = 'emov_db.csv'
data_stat_path = 'data_stat'

def on_progress(p):
    for k,v in p.items():
        logging.debug("%s: %s" % (k, v))


def load_emov_db(path_to_EmoV_DB=None, load_csv=False):
    if load_csv or os.path.exists(emo_csv_path):
        data = load_csv_db()
        print("DB loaded from {} !".format(emo_csv_path))

        return data

    print('Start to load wavs.')

    transcript = os.path.join(path_to_EmoV_DB, 'cmuarctic.data')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()

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
                fpath = os.path.join(path_to_EmoV_DB, spk, emo, file)

                if file[-4:] == '.wav':
                    fnumber = file[-8:-4]
                    #print(fnumber)
                    if fnumber.isdigit():
                        text = sentences[sentences['n'] == fnumber]['text'].iloc[0]  # result must be a string and not a df with a single element
                        # text_lengths.append(len(text))
                        # texts.append(text)
                        # texts.append(np.array(text, np.int32).tostring())
                        # fpaths.append(fpath)
                        # emo_cats.append(emo)

                        duration = get_wav_duration(fpath)

                        e = {'database': 'EmoV-DB',
                             'id': file[:-4],
                             'speaker': spk,
                             'emotion':emo,
                             'transcription': text,
                             'sentence_path': fpath,
                             'duration': duration}
                        data.append(e)
                        #print(e)

    data = pd.DataFrame.from_records(data)

    #data = fix_wrong_transcriptions(data)

    print("DB loaded from {} !".format(path_emov_db))

    return data

def fix_wrong_transcriptions(data):
    condition = (data.speaker == 'sam') & (data.emotion == 'disgusted') & (data.id == 'disgusted_57-84_0075')
    data.loc[condition, 'transcription'] = 'The gray eyes faltered; the flush deepened.'

    condition = (data.speaker == 'josh') & (data.emotion == 'amused') & (data.id == 'amused_225_252_0026')
    data.loc[condition, 'transcription'] = 'I came before my ABCs.'

    return data


def align_db(data):
    import pathlib
    except_i_list = list(range(len(data)))

    while True:
        for i in tqdm(except_i_list):
            row = data.iloc[i]
            f = row.sentence_path
            transcript = row.transcription
            with gentle.resampled(f) as wavfile:
                aligner = gentle.ForcedAligner(resources, transcript, nthreads=40)
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
    import pathlib

    splitted_path = split_path(align_json_path)
    json_file_name = splitted_path[-1]
    id, _ = os.path.splitext(json_file_name)
    emotion = splitted_path[-2]
    speaker = splitted_path[-3]


    row = data[(data.id == id) & (data.speaker == speaker)].iloc[0] # iloc: df to series
    f = row.sentence_path
    transcript = row.transcription
    with gentle.resampled(f) as wavfile:
        aligner = gentle.ForcedAligner(resources, transcript, nthreads=40)
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


# path='alignments/EmoV-DB/bea/amused/amused_1-15_0001.json'
# start, end=get_start_end_from_json(path)

def play_start_end(path, start, end):
    import sounddevice as sd

    import librosa

    y,fs=librosa.load(path)
    sd.play(y[int(start*fs):int(end*fs)],fs)


def save_wav_start_end(ori_wav_path, new_wav_path, start, end):
    import librosa

    y,fs=librosa.load(ori_wav_path)
    #librosa.output.write_wav(new_wav_path, y[int(start*fs):],fs)
    # End time are wrong often.
    if math.isnan(end):
        librosa.output.write_wav(new_wav_path, y[int(start*fs):],fs)
    else:
        librosa.output.write_wav(new_wav_path, y[int(start*fs):int(end*fs)],fs)


def get_wav_duration(wav_path):
    import librosa

    y, fs = librosa.load(wav_path)
    n_frames = len(y)
    seconds =  float(n_frames) / fs

    return seconds

def play(path):
    import sounddevice as sd

    import librosa

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


def trim_wavs_with_start_end():
    error_json_path_list = list()
    data = load_emov_db(path_emov_db)
    for i, row in tqdm(data.iterrows(), total=len(data)):
        original_wav_path = row.sentence_path
        o_wav_dir, o_wav_name = os.path.split(original_wav_path)
        #json_path = 'alignments/EmoV-DB_sorted/bea/Amused/amused_1-15_0001.json'
        try:
            json_path = os.path.join('alignments', 'EmoV-DB_sorted', row.speaker, row.emotion, '{}.json'.format(row.id))
            start, end = get_start_end_from_json(json_path)

            if row.emotion != 'neutral':
                save_wav_start_end(original_wav_path, original_wav_path, start, end)

        except Exception as e:
            print(e)
            error_json_path_list.append(json_path)
    print('#Errors:', len(error_json_path_list))
    pprint(error_json_path_list)
    with open('trim_except_json_path_list.txt', 'w') as f:
        f.write(str(error_json_path_list))


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
        emo_dir = emo_dir.lower()

        old_path = row.sentence_path
        new_path = os.path.join(emo_dir_dir_wav, emo_dir, filename_wav)
        print('From:', old_path)
        print('To  :', new_path)

        new_dir = os.path.join(emo_dir_dir_wav, emo_dir)
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

        old_path = path_json
        new_path = os.path.join(emo_dir_dir_json, emo_dir, filename_json)

        print('From:', old_path)
        print('To  :', new_path)
        print()

        new_dir = os.path.join(emo_dir_dir_json, emo_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        os.rename(
            old_path,
            new_path
        )

def replace_file_name(string):
    string = string.replace('sleepiness', 'sleepy')
    string = string.replace('disgust', 'disgusted')
    string = string.replace('anger', 'angry')
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


def save_db_to_csv():
    data = load_emov_db(path_emov_db)
    data.to_csv(emo_csv_path, index=False)
    data.to_csv(os.path.join(path_emov_db, emo_csv_path), index=False)
    data.to_csv(os.path.join(data_stat_path, emo_csv_path), index=False)


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
    df_by_spk= df.groupby(['speaker', 'emotion'])['duration'].agg(['size','min', 'max', 'mean', 'std', 'sum'])
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
    #make_alignments()
    #align_again_nan_start_end()
    #check_wrong_alignments()
    #trim_wavs_with_start_end()
    save_db_to_csv()
    get_db_stat()

    pass
