#####################################################
# Yuval PM 21/10
# function to display or save images, movies.
#####################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

WHITE = 255
FRAME_ROWS = 896
FRAME_COLS = 900

def show_image(im):
    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()
    return im

def display_gray_movie(images):

    fig, ax = plt.subplots()
    ims = []

    for frame in images:
        im = ax.imshow(frame, cmap=plt.cm.gray, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                    repeat_delay=1000)

    # ani.save('dynamic_images.mp4')
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def save_gray_movie(event, output_path):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (FRAME_ROWS, FRAME_COLS))
    for i, frame in enumerate(event):

        if frame.max() == 1:
            frame = frame*WHITE
        frame_8 = frame.astype('uint8')
        color_frame = cv2.cvtColor(frame_8, cv2.COLOR_GRAY2BGR)

        out.write(color_frame)
    out.release()

def save_color_movie(event, output_path):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (FRAME_ROWS, FRAME_COLS))
    for i, frame in enumerate(event):
        frame_8 = frame.astype('uint8')
        out.write(frame_8)
    out.release()

def from_params_data_to_movie(params_list, orig_event, output_path):
    event_marked_param = []
    for para in params_list:
        para.location = [None]*para.timestamp + para.location

    for i, frame in enumerate(orig_event):
        frame_marked_param = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for para in params_list:
            if i < len(para.location):
                if not para.location[i] is None:
                    cv2.circle(frame_marked_param, (int(para.location[i][1]), int(para.location[i][0])), 3, para.color, -1)


        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(frame_marked_param, str(i), org, font, fontScale, color, thickness, cv2.LINE_AA)
        event_marked_param.append(frame_marked_param)

    save_color_movie(event_marked_param, output_path)
    return
