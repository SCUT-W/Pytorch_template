import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import librosa.display
import random
sr=44100
n_mels=128
hop_length = int(0.01 * sr)
win_length = int(0.06 * sr)

def process_audio_to_log_melspectrogram(audio_path, target_duration=30.0, offset=15.0,n_fft=win_length, hop_length=hop_length):
    y, sr = librosa.load(audio_path, sr=44100,offset=offset)
    target_length = int(target_duration * sr)
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = librosa.util.fix_length(y, size=target_length)
    step_size=int(0.5*sr)
    mel_spectrograms = []
    begin=int(0.0*sr)
    for start in range(begin, target_length, step_size):
        end = start + step_size
        if end > target_length:
            end = target_length
        segment = y[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(
            y=segment, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length
        )
        log_mel_spectrogram= np.log(
            mel_spectrogram + 1e-6
        )
        mel_spectrograms.append(log_mel_spectrogram[:,:-1])
    log_mel_spectrogram=np.array(mel_spectrograms)
    log_mel_spectrogram=np.transpose(log_mel_spectrogram, axes=(0,2,1))
    return log_mel_spectrogram

def get_audio_label(label_file_path,audio_name_list):
    df=pd.read_csv(label_file_path)
    label_list=[]
    for audio_name in audio_name_list:
        print("Label Processing",audio_name)
        label=df[df['song_id']==audio_name]
        label_list.append(label.iloc[0, 1:-1].tolist())
    label_array=np.array(label_list)
    return label_array

def process_set_of_music(audio_name_list,annotation_a_path,annotation_v_path,target_duration=30.0, offset=15.0,n_fft=win_length, hop_length=hop_length,training=True):
    input_directory='../data/1000songs/raw_data/'

    if not training:
        target_duration=30.0 #Need to fix
        target_directory_spec='../data/1000songs/test_data/log_mel_spec.npy'
        target_directory_label_a='../data/1000songs/test_data/label_a.npy'
        target_directory_label_v='../data/1000songs/test_data/label_v.npy'
    else:
        target_directory_spec='../data/1000songs/train_data/log_mel_spec.npy'
        target_directory_label_a='../data/1000songs/train_data/label_a.npy'
        target_directory_label_v='../data/1000songs/train_data/label_v.npy'


    #Process spectrogram
    spectrogram_list=[]
    for audio_name in audio_name_list:
        print("Spectrogram Processing",audio_name)
        audio_path=os.path.join(input_directory,f"{audio_name}.wav")
        spectrogram=process_audio_to_log_melspectrogram(audio_path,target_duration, offset, n_fft, hop_length)
        spectrogram_list.append(spectrogram)
    spectrogram_list=np.array(spectrogram_list)
    spectrogram_list=np.expand_dims(spectrogram_list,axis=2)
    print("Final spectrogram shape", spectrogram_list.shape)
    np.save(target_directory_spec,spectrogram_list)

    #Process label a
    label_a=get_audio_label(annotation_a_path,audio_name_list)
    print("Final label shape", label_a.shape)
    np.save(target_directory_label_a,label_a)

    #Process label v
    label_v=get_audio_label(annotation_v_path,audio_name_list)
    np.save(target_directory_label_v,label_v)

def mp32wav(input_folder, output_folder, target_sample_rate=44100):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp3"):
            mp3_path = os.path.join(input_folder, file_name)
            wav_path = os.path.join(output_folder, file_name.replace(".mp3", ".wav"))
            y, sr = librosa.load(mp3_path, sr=target_sample_rate)
            sf.write(wav_path, y, sr)
            print("Finished processing {}".format(mp3_path))

def ensure_folders_exist(folder_list):
    for folder in folder_list:
        if not os.path.exists(folder):
            os.makedirs(folder)

def initial_preprocess(dataset_name):
    folders = [
        f"../data/{dataset_name}",
        f"../data/{dataset_name}/train_data",
        f"../data/{dataset_name}/test_data",
        f"../data/{dataset_name}/raw_data",
        f"../data/{dataset_name}/annotation"
    ]
    ensure_folders_exist(folders)
def spilt_train_test(annotation_file_path,test_ratio,random_seed=42):
    '''
    Randomly split the dataset into train and test sets.
    Return sets (x.wav)
    '''
    # file_list=[]
    # for f in os.listdir(raw_data_folder):
    #     file_list.append(f)
    df=pd.read_csv(annotation_file_path)
    file_list=df['song_id'].tolist()
    np.random.seed(random_seed)
    np.random.shuffle(file_list)
    train_file_name_list=file_list[:int((1-test_ratio)*len(file_list))]
    test_file_name_list=file_list[int((1-test_ratio)*len(file_list)):]
    return train_file_name_list, test_file_name_list


if __name__ == '__main__':
    initial_preprocess("1000songs")
    input_folder=r"C:\Users\89721\Desktop\clips_45sec\clips_45seconds"
    output_folder='../data/1000songs/raw_data'
    annotation_a_file_path='../data/1000songs/annotation/arousal_cont_average.csv'
    annotation_v_file_path='../data/1000songs/annotation/valence_cont_average.csv'
    if len(os.listdir('../data/1000songs/raw_data'))==0:
        mp32wav(input_folder,output_folder)
    train_set,test_set=spilt_train_test(r"C:\Users\89721\Desktop\annotations\arousal_cont_average.csv",test_ratio=0.2)
    process_set_of_music(train_set,annotation_a_file_path,annotation_v_file_path,training=True)
    process_set_of_music(test_set,annotation_a_file_path,annotation_v_file_path,training=False)








