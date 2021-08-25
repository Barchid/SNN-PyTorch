# converts an input image to a video of the same image, but moving with saccadic motion (similarly to NMIST, N-Caltech, etc). The image will be 300ms long
import cv2
import argparse
import numpy as np


def translate(image: np.ndarray, t_x: float, t_y: float):
    shifted = cv2.warpAffine(image, np.float32([
        [1, 0, t_x],
        [0, 1, t_y]
    ]), (image.shape[1], image.shape[0]))
    return shifted


def save_crop(image: np.ndarray, output: cv2.VideoWriter, height: int, width: int):
    # crop the image to obtain the frame of the video
    center = (image.shape[0] // 2, image.shape[1] // 2)
    top_left_crop = (center[0] - height//2, center[1] - width//2)
    bot_right_crop = (center[0] + height//2, center[1] + width//2)

    frame = image[top_left_crop[0]:bot_right_crop[0],
                  top_left_crop[1]:bot_right_crop[1]]

    # save output video
    output.write(frame)


def saccade(image, video, args, has_rest=True):
    # duration of the three saccades (in ms)
    d = 100.

    fps = 100.
    ms_per_frame = 1000./fps  # number of ms for one frame in the output video
    frame_d = d/ms_per_frame  # number of frames in the saccade

    # Frame rate of
    # vertical and horizontal distances
    t = 10.

    dt = t // frame_d

    save_crop(image, video, args.height, args.width)
    # SACCADE 1 (UP-LEFT TO LOW) during 100ms
    for _ in range(int(dt)):
        # translate
        image = translate(image, 1, 1)
        save_crop(image, video, args.height, args.width)

    # SACCADE 2 (LOW TO UP-RIGHT)
    for _ in range(int(dt)):
        # translate
        image = translate(image, 1, -1)
        save_crop(image, video, args.height, args.width)

    # SACCADE 3 (UP-RIGHT TO UP-LEFT)
    for _ in range(int(dt)):
        image = translate(image, -2, 0)
        save_crop(image, video, args.height, args.width)

    # REST after saccade
    for _ in range(int(3*dt)):
        save_crop(image, video, args.height, args.width)


def main():
    args = get_args()

    image = cv2.imread(args.path, cv2.IMREAD_COLOR)

    video = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 100, (args.width, args.height))

    for _ in range(50):
        saccade(image, video, args, has_rest=args.rest)

    print('Saving result')
    video.release()


def get_args():
    parser = argparse.ArgumentParser(description='Image to saccade script')
    parser.add_argument('--path', type=str,
                        help="Path of the image.", required=True)
    parser.add_argument('--output', '-o', type=str,
                        help="Path of output video.", default="output.mkv")
    parser.add_argument('--height', type=int, default=128,
                        help="Height of the output video")
    parser.add_argument('--width', type=int,
                        default=128, help="width of the output video")
    parser.add_argument('--rest', '-r', action='store_true', default=False,
                        help="Flag that indicates whether there is a rest time of 100ms after the saccade")

    return parser.parse_args()


if __name__ == '__main__':
    main()
