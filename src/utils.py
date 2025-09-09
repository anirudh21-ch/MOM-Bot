import matplotlib.pyplot as plt
import numpy as np

def read_file(path_to_file: str):
    """
    Read a text file into a list of lines.
    """
    with open(path_to_file) as f:
        return f.read().splitlines()


def display_waveform(signal, sample_rate, text='Audio', overlay_color=None):
    """
    Plot an audio waveform, with optional overlay coloring.
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 2))
    plt.scatter(np.arange(len(signal)), signal, s=1, c='k')
    if overlay_color:
        plt.scatter(np.arange(len(signal)), signal, s=1, c=overlay_color)
    plt.title(text, fontsize=16)
    plt.xlabel('time (secs)', fontsize=18)
    plt.ylabel('signal strength', fontsize=14)
    plt.axis([0, len(signal), -0.5, 0.5])
    time_axis, _ = plt.xticks()
    plt.xticks(time_axis[:-1], time_axis[:-1] / sample_rate)
    plt.show()


def get_color(signal, speech_labels, sample_rate=16000):
    """
    Map speech segments to color-coded labels for visualization.

    Args:
        signal: Audio signal array.
        speech_labels: List of diarization labels [start, end, speaker].
        sample_rate: Sample rate of the signal.

    Returns:
        list of colors for each sample in the signal.
    """
    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
              'magenta', 'cyan', 'pink', 'gray']
    c = ['black'] * len(signal)
    for item in speech_labels:
        try:
            start = int(float(item[0]) * sample_rate)
            end = int(float(item[1]) * sample_rate)
            speaker_idx = int(item[2].split('_')[-1])
            code = COLORS[speaker_idx % len(COLORS)]
            c[start:end] = [code] * (end - start)
        except Exception as e:
            print(f"Skipping segment {item} due to error: {e}")
            continue
    return c
