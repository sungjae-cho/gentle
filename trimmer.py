import webrtcvad, os, wave, contextlib, collections, argparse
import librosa
from tqdm import tqdm

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    output = []
    while offset + n < len(audio):
        output.append(Frame(audio[offset:offset + n], timestamp, duration))
        timestamp += duration
        offset += n
    return output

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, frames, filename):
    sample_width = 2
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    aggressiveness = 3
    while aggressiveness >= 0:
        vad = webrtcvad.Vad(aggressiveness)
        voiced_frames = []
        trigger_indexes = []
        va_time_start_list = []
        va_time_end_list = []
        i_frame = 0
        for frame in frames:
            if not triggered:  #unvoiced part
                ring_buffer.append(frame)
                num_voiced = len([f for f in ring_buffer
                                  if vad.is_speech(f.bytes, sample_rate)])
                if num_voiced > 0.3 * ring_buffer.maxlen:
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
                    trigger_indexes.append(i_frame)
                    triggered_time = i_frame * frame_duration_ms / 1000
                    va_time_start_list.append(triggered_time)
            else:  #voiced part
                voiced_frames.append(frame)
                ring_buffer.append(frame)
                num_unvoiced = len([f for f in ring_buffer
                                    if not vad.is_speech(f.bytes, sample_rate)])
                if num_unvoiced > 0.3 * ring_buffer.maxlen:
                    triggered = False
                    ring_buffer.clear()
                    trigger_indexes.append(i_frame)
                    triggered_time = i_frame * frame_duration_ms / 1000
                    va_time_end_list.append(triggered_time)
            i_frame += 1
        if voiced_frames:
            time_start = min(va_time_start_list)
            if len(va_time_start_list) == len(va_time_end_list):
                time_end = max(va_time_end_list)
            else:
                time_end = len(frames) * frame_duration_ms / 1000
            return b''.join([f.bytes for f in voiced_frames]), time_start, time_end, va_time_start_list, va_time_end_list
        aggressiveness -= 1
    print('Could not find voice activity at', filename)
    time_end = len(frames) * frame_duration_ms / 1000
    return b''.join([f.bytes for f in frames]), 0, time_end, va_time_start_list, va_time_end_list

def trim(rDirectory, wDirectory):
    frame_duration_ms = 30
    padding_duration_ms = 300
    # print("read dir: ", rDirectory)
    for root, dirnames, filenames in os.walk(rDirectory):
        for filename in tqdm(filenames):
            if filename[-4:] == '.wav':
                rf = os.path.join(root, filename)
                dir_speaker, dir_emo = os.path.split(root)
                dir_db, dir_speaker = os.path.split(dir_speaker)
                audio, sample_rate = read_wave(rf)
                #y, fs = librosa.load(rf, sample_rate)
                #print("Seconds:", len(audio)  / sample_rate / 2)
                frames = frame_generator(frame_duration_ms, audio, sample_rate)
                segment, time_start, time_end, time_start_list, time_end_list = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, frames, rf)
                #print("time_start:", time_start)
                #print("time_end:", time_end)
                wPath = os.path.join(wDirectory, dir_speaker, dir_emo, filename)

                write_wave(wPath, segment, sample_rate)

def get_vad_ranges(wav_path, frame_duration_ms=30, padding_duration_ms=300):
    audio, sample_rate = read_wave(wav_path)
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    segment, time_start, time_end, time_start_list, time_end_list = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, frames, wav_path)

    return time_start, time_end, time_start_list, time_end_list



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--in_dir', type=str, help='type dataset for trimming')
    parser.add_argument('--out_dir', type=str, help='type dataset for trimming')
    args = parser.parse_args()

    if not args.in_dir or not args.out_dir:
        parser.error('--in_dir and --out_dir should be given')

    in_dir = args.in_dir
    out_dir = args.out_dir

    # ------ trimming scilence using VAD
    os.makedirs(out_dir, exist_ok=True)
    trim(in_dir, out_dir)
