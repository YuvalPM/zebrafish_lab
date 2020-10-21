#####################################################
# Yuval PM 19/10
# create binary event
#####################################################

from scipy import stats
from visualize import *
import os
import numpy as np

WHITE = 255
FRAME_ROWS = 896
FRAME_COLS = 900


def make_noise(random_frames_dir, noise_frame_path, has_np_random_frames, np_random_frames_path):
    '''
    step 1:
        create numpy array contains random frames from a directory of random frames of the same fish.
    step 2:
        create a frame represents the mean+3*sd shade of each pixel.
        this is the 'static noise' frame.
    :param random_frames_dir: directory path, contains the random frames of the fish video.
    :param noise_frame_path: file path, to save the noise frame numpy array
    :param has_np_random_frames: boolean, if there is a saved numpy array contains the random frames,
    is can save time to not create a new one.
    :param np_random_frames_path:  file path, to save/load numpy array contains the random frames.
    :return: return a numpy array represent the noise frame and save it as a .npy file
    '''

    print("- - - - make noise step 1 - - - -")

    if has_np_random_frames:
        noise = np.load(np_random_frames_path)
    else:
        frames_list = os.listdir(random_frames_dir)
        noise = np.zeros([len(frames_list), FRAME_COLS, FRAME_ROWS])
        i = 0
        for file_frame in frames_list:
            if i % 200 == 0:
                print(i)

            frame = np.fromfile(random_frames_dir + '\\' + file_frame, dtype=np.uint8)
            frame = frame.reshape([FRAME_COLS, FRAME_ROWS])
            noise[i, :, :] = frame
            i += 1

        np.save(np_random_frames_path, noise)

    print("- - - - make noise step 2 - - - -")
    mPixelVal = np.zeros([FRAME_COLS, FRAME_ROWS])
    sdPixelVal = np.zeros([FRAME_COLS, FRAME_ROWS])
    for i in range(FRAME_ROWS):
        for j in range(FRAME_COLS):
            if j % 200 == 0:
                print("pixel: " + str(i) + "," + str(j))
            fit_normal = stats.norm.fit(noise[:, j, i])
            mPixelVal[j, i] = fit_normal[0]
            sdPixelVal[j, i] = fit_normal[1]
            # the same:
            # mPixelVal[j, i] = np.mean(noise[:, j, i])
            # stdPixelVal[j, i] = np.std(noise[:, j, i])

    only_noise = (mPixelVal + (3 * sdPixelVal))
    np.save(noise_frame_path, only_noise)
    return only_noise


def clean_frame(dirty, only_noise):
    '''
    clean the noise of sample frames of the whole video, from the frame we want to 'clean'
    the input frames have the same shape.
    :param dirty: frame as numpy array
    :param only_noise: frame as numpy array
    :return: the clean frame as numpy array
    '''

    clean = dirty.astype(float) - only_noise
    clean = np.clip(clean, 0, WHITE)
    return clean


def clean2binary(clean, option, fixed_th=0):
    '''
    make it binary with different threshold
    option 0: no additional threshold
    option 1: looking for the next pick in the histogram (after the median)
    option 2: looking for the next pick in the histogram (after 0)
    option 3: a fixed threshold from a given input
    :param clean: a single frame without static noise
    :param option: which threshold to calculate
    :param fixed_th: used only in option 3
    :return: binary frame
    '''
    threshold = 0

    if option == 0:
        threshold = 0

    if option == 1:
        hist = np.histogram(clean, bins=257)[0]
        threshold = int(round(float(np.median(clean))))
        while hist[threshold] > hist[threshold + 1] or hist[threshold] > hist[threshold + 2]:
            threshold += 1
            if threshold == 256:
                break

    if option == 2:
        hist = np.histogram(clean, bins=257)[0]
        threshold = 0
        while hist[threshold] > hist[threshold + 1] or hist[threshold] > hist[threshold + 2]:
            threshold += 1
            if threshold == 256:
                break

    if option == 3:
        threshold = fixed_th

    bin_frame = (clean > threshold).astype(int)
    # print(threshold)
    return threshold, bin_frame


def make_bin_event(orig_event, noise):
    '''
    step 1: looking for the best threshold for each frame, and save the maximal threshold.
    step 2: binaryze each frame with the maximal threshold
    :param orig_event: numpy array of the event (array of frames)
    :param noise: numpy array of the noise (single frame)
    :return: numpy array of the bin event (array of frames)
    '''
    print("- - - - make binary event step 1 - - - -")
    max_th = 0
    clean_list = np.zeros(orig_event.shape)
    bin_list = np.zeros(orig_event.shape)

    # step 1:
    # looking for the best threshold for each frame, and save the maximal threshold.
    for i, frame in enumerate(orig_event):
        if i % 100 == 0:
            print(i)
        clean = clean_frame(frame, noise)
        clean_list[i, :, :] = clean
        th, binary = clean2binary(clean, 2)
        if max_th < th:
            max_th = th

    print("- - - - make binary event step 1 - - - -")
    # step 2:
    # binarize each frame with the maximal threshold
    for j, clean in enumerate(clean_list):
        if j % 100 == 0:
            print(j)
        th, binary = clean2binary(clean, 3, max_th)
        bin_list[j, :, :] = binary
    return bin_list


if __name__ == '__main__':
    random_frames_dir = 'E:\\Lab-Shared\\Data\\FeedingAssay2020\\20200720-f3-2000'
    np_random_frames_path = '..\\output_np\\random_noise.npy'
    np_noise_frame_path = '..\\output_np\\noise_frame.npy'
    event_path = 'E:\\Lab-Shared\\Data\\FeedingAssay2020\\20200720-f3\\20200720-f3-9.raw'
    bin_event_path = '..\\output_np\\binary_events\\20200720-f3-7.npy'

    # if there is no 'noise' frame, create one. The frame is saved as numpy array
    if os.path.exists(np_noise_frame_path):
        noise_frame = np.load(np_noise_frame_path)
    else:
        noise_frame = make_noise(np_noise_frame_path, random_frames_dir, True, np_random_frames_path)

    # load the raw_date into numpy array
    orig_event = np.fromfile(event_path, dtype=np.uint8)
    orig_event = np.reshape(orig_event, [orig_event.size // (FRAME_ROWS * FRAME_COLS), FRAME_COLS, FRAME_ROWS])

    # create a new event that contains binary values.
    bin_event = make_bin_event(orig_event, noise_frame)

    # to save as a numpy array
    np.save(bin_event_path, bin_event)
    # to display the numpy array
    display_gray_movie(orig_event)
    display_gray_movie(bin_event)