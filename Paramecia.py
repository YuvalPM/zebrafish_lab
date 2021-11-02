

from visualize import from_params_data_to_movie, mark_points_movie, save_gray_movie, show_image
from matplotlib import pyplot as plt
import numpy as np
import pickle
from imageio import imread, imwrite
from itertools import permutations


from collections import deque
from find_objects import find_params_location, find_param, find_fish, separate_fish_param, predict2d
from make_binary import make_bin_event

FRAME_ROWS = 896
FRAME_COLS = 900


def euclidean_distance(a, b):
    dist = np.sqrt(np.sum(np.power((a - b), 2), axis=1))
    return dist

def find_perfect_match_equel(list_a, list_b):
    best_permute = list_b
    min_score = 100
    np_a = np.array(list_a)

    all_permutations = permutations(list_b)
    for permute in all_permutations:
        dist = euclidean_distance(np_a, permute)
        score = sum(dist)
        if score < min_score:
            min_score = score
            best_permute = permute

    return min_score, best_permute


def find_perfect_match(list_a, list_b):
    perfect = {}
    if len(list_a) == len(list_b):
        best_permute_a = list_a
        score, best_permute_b = find_perfect_match_equel(list_a, list_b)

    else:
        if len(list_a) < len(list_b):
            len_a = len(list_a)
            best_permute_a = list_a
            best_permute_b = list_b[:len_a]
            min_score = 100
            for i in range(len(list_b)-len(list_a)):
                permute_b = list_b[i:len_a+i]
                score, permute_b = find_perfect_match_equel(list_a, permute_b)
                if score < min_score:
                    min_score = score
                    best_permute_b = permute_b
        else:
            len_b = len(list_b)
            best_permute_b = list_b
            best_permute_a = list_a[:len_b]
            min_score = 100
            for i in range(len(list_a) - len(list_b)):
                permute_a = list_a[i:len_b + i]
                score, permute_a = find_perfect_match_equel(list_b, permute_a)
                if score < min_score:
                    min_score = score
                    best_permute_a = permute_a

    for a, b in zip(best_permute_a, best_permute_b):
        perfect[a] = b
    return perfect


class Para:
    """
    This class represents a single Paramecia.
    It holds information and history for the entire event.
    The purpose is to keep track of a specific paramecia while dealing some of these challenges:
    1. when there are two (or more) paramecia in the next frames that can be good candidate to be this current one.
    2. there are no good candidate for this paramecia in the next frame: collision of two paramecium so they look
     like one, collision of this paramecia with the fish, collision of this paramecia with the plate, or too small
     paramecia so it looks like a dust. (due too threshold for the binarization of the event or threshold for the
     size of a paramecia.)

     In most of the frames, the paramecia's location is determined by the white spots in the image,
     but when the paramecia disapear for a few frames, it determind by the last location or by a simple prediction.
    """
    ids = 0
    FISH = 0
    PARA = 1
    ELSE = 2
    FROM_IMG = 0
    REPEAT_LAST = 1
    PREDICT = 2
    PREDICT_AND_IMG = 3
    DOUBLE_PARA = 4

    def __init__(self, timestmp, coord, region):
        self.id = Para.ids
        Para.ids += 1
        self.location = [coord]
        self.lastcoord = coord
        self.timestamp = timestmp
        self.color = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
        self.completed = False
        self.completedstmp = -1
        self.waitindex = 0
        # waits at max 4 frames before giving up on particular para.
        # todo: is it the best max?
        self.waitmax = 8

        # todo: is it the best max?
        self.thresh = 5

        self.fish_thresh = 12

        self.double_thresh = 10
        self.double = False
        self.num_double = 1

        self.region_list = [region]
        self.all_pixel_list = [region.coords.astype(np.float)]
        self.partial_pixel_list = [region.coords]
        self.certainty = [self.FROM_IMG]
        self.predict_x = None
        self.predict_y = None
        self.predict_x_direction = None
        self.predict_y_direction = None

    def end_record(self, timestmp):
        # CHOPS OFF THE HANGING LEN(WAITMAX) PORTION WHERE IT COULD'NT FIND PARTNER
        self.location = self.location[:-self.waitmax]
        self.region_list = self.region_list[:-self.waitmax]
        self.all_pixel_list = self.all_pixel_list[:-self.waitmax]
        self.partial_pixel_list = self.partial_pixel_list[:-self.waitmax]
        self.certainty = self.certainty[:-self.waitmax]
        self.completed = True
        self.completedstmp = timestmp-self.waitmax

    def nearby(self, cont_list, completed_list, timestmp, double_options):
        """
        :param cont_list: list of contours/regions found in the image, represents potential paramecium.
       The object region contains geometric information about the contour, and about the location in the given image
        as returned in the function skimage.measure.regionprops.
       Documentation: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
        :param completed_list: list of params that are not found for 8 frames and are not recorded any more.
        :param timestmp: current frame
        :param double_options: empty list to fill with paramecium marked as double
        :return:
        """

        # first, search the list of contours found in the image to see if any of them are near
        # the previous position of this Para.
        # loc_list is a list of tuples represents the center point of the paramecium
        loc_list = [para.centroid for para in cont_list]

        # second, filter only the short distance contours to be the potential next frame paramecia,
        # and save those in np.array
        if len(cont_list) == 0:
            para_center_coords = np.array([])
            para_regions = np.array([])
        else:
            loc_arr = np.array(loc_list)
            cont_arr = np.array(cont_list)
            dist = euclidean_distance(loc_arr, np.array(self.lastcoord))

            para_center_coords = loc_arr[dist < self.thresh]
            para_regions = cont_arr[dist < self.thresh]

        # if there's nothing found, add 1 to the waitindex, and say current position is the last position
        if para_center_coords.shape[0] == 0:

            self.location.append(self.lastcoord)
            self.region_list.append(self.region_list[-1])
            self.all_pixel_list.append(self.all_pixel_list[-1])
            self.partial_pixel_list.append(self.all_pixel_list[-1])
            self.certainty.append(self.REPEAT_LAST)

            if self.waitindex == self.waitmax:
                # end record if you've gone 'waitmax' frames without finding anything.
                # this value greatly changes things. its a delicate balance between losing the para and waiting too
                # long while another para enters
                self.end_record(timestmp)
            self.waitindex += 1

        # this case is only one contour is within the threshold distance to this Para.
        elif para_center_coords.shape[0] == 1:

            new_coord = para_center_coords[0]
            new_reg = para_regions[0]
            self.update(new_coord, new_reg, self.FROM_IMG)
            cont_list.remove(new_reg)

        # This case is when two or more contours fit threshold distance.
        # That means that maybe the paramecia is actually two or more paramecium after collision so it looked like
        # there is only one until now. Try to match between paramecium that were stop recorded potentially because
        # of this collision, and between new paramecium that found in this current frame.
        elif para_center_coords.shape[0] > 1:
            # for now this information is not used. But we should use it to better tracking
            self.double = True

            # create list of close 'doubles'
            after_potential = para_regions
            before_potential = self.find_close_before_double(completed_list, timestmp)
            para_reg_match = self.match_double(before_potential, after_potential, timestmp) # para_reg_match is a dictionary
            for para in para_reg_match:
                completed_list.remove(para)
                double_options.append(para)
                cont_list.remove(para_reg_match[para])

        return cont_list

    def match_double(self, before, after, cur_frame):
        """
        This function matches between old paramecium that meybe collided with this (self) paramecia, and between new
        paramecium that are now very close to this (self) paramecia, potentially just finished the 'collision'.
        For every matched pair of 'before' and 'after' paramecium the function merges their data as one paramecia,
        and fill in the missing data with this (self) paramecia data as they where double paramecium tracked together.

        :param before: List of paramecium that where close to the paramecia when completed the record
        :param after: List of regions represents potential paramecium after the collision
        :param cur_frame: The frame where more then one paramecium appeared close the a single paramecia
        :return: Dictionary with key- Para object of a 'before' paramecia, and value- region object of
         an 'after' paramecia.
        """
        predicted_location = []
        for para in before+[self]:
            para.create_prediction()
            x = para.predict_x(cur_frame)
            y = para.predict_y(cur_frame)
            predicted_location.append((x, y))

        real_location = [reg.centroid for reg in after]

        # perfect_match is a dictionary with key- tuple represents the predicted center of a 'before' paramecia,
        # and value- tuple represents the real center of an 'after' paramecia.
        perfect_match = find_perfect_match(predicted_location, real_location)
        para_reg_match = {}
        for predict in perfect_match:
            real_coord = perfect_match[predict]
            real_index = real_location.index(real_coord)
            real_reg = after[real_index]

            para_index = predicted_location.index(predict)
            para = (before + [self])[para_index]

            if para.completed:
                para_reg_match[para] = real_reg
                para.location += self.location[para.completedstmp:]
                para.region_list += self.region_list[para.completedstmp:]
                para.all_pixel_list += self.all_pixel_list[para.completedstmp:]
                para.certainty += [self.DOUBLE_PARA]*(cur_frame-para.completedstmp)
            para.update(real_coord, real_reg, self.FROM_IMG)
        return para_reg_match

    def find_close_before_double(self, completed_list, cur_frame):
        """
        creates a list of all the paramecium that at the frame they where last tracked, they were close to this
         paramecia and potentially collided, so they looked like one paramecia.
        :param completed_list: list of paramecium that are not recorded anymore
        :param cur_frame:
        :return: list of paramecium that are not recorded anymore and where close to this paramecia when last recorded.
        """
        close_before_double = []
        for para in completed_list:
            frame_before = cur_frame-para.completedstmp
            if frame_before > len(self.location) or frame_before < 0:
                continue
            dist = euclidean_distance(np.array([para.lastcoord]), self.location[-frame_before])
            if dist[0] < self.double_thresh:
                close_before_double.append(para)

        return close_before_double

    def param_trajectory(self):
        """
        plot the trajectory of the paramecia
        :return:
        """
        x = [loc[1] for loc in self.location]
        y = [loc[0] for loc in self.location]

        fig, ax = plt.subplots()
        ax.plot(x, y, '--g', linewidth=0.2)
        plt.tight_layout()
        plt.show()


    def fish_collision(self, fish):
        """
        Finds collision points with the fish
        :param fish_coords: ndarray of tuples represents all the fish's pixels.
        :return: tuple of the collision point
        """
        fish_coords = fish.coords
        dist = euclidean_distance(fish_coords, np.array(self.lastcoord))

        if min(dist) < self.fish_thresh:  # todo find the best threshold.
            return fish_coords[dist == min(dist)][0]
        else:
            return None


    def create_prediction(self):
        """
        assume the paramecia trajectory is a straight line approximately, predict location and direction
        :return:
        """
        fitt_from = (len(self.location) // 3) * 2
        predict_x, predict_y, x_direction, y_direction = predict2d(np.arange(fitt_from, len(self.location)), self.location[fitt_from:])
        self.predict_x = predict_x
        self.predict_y = predict_y
        self.predict_x_direction = x_direction
        self.predict_y_direction = y_direction

    def predict_update(self, cur_frame, fish):
        """
         fill in the missing the data of a paramecia using prediction, from its last location until current frame
         :param cur_frame: the most current frame to fill the predicted locarion
         :param fish_coords: fish's pixels
         :return:
         """
        for i in range(len(self.location), cur_frame - self.timestamp):  # complete only missing relative to timestamp):
            x_predict = self.predict_x(i)
            y_predict = self.predict_y(i)
            new_pixels = np.array(self.all_pixel_list[i-1]).astype(np.float)
            new_pixels[:, 0] = np.array(self.all_pixel_list[i-1])[:, 0] + self.predict_x_direction
            new_pixels[:, 1] = np.array(self.all_pixel_list[i-1])[:, 1] + self.predict_y_direction
            self.lastcoord = (x_predict, y_predict)
            self.location.append(self.lastcoord)
            self.region_list.append(None)
            self.all_pixel_list.append(new_pixels)
            new_partial = self.predict_visible_pixels(fish, np.array(self.partial_pixel_list[-1]))
            self.partial_pixel_list.append(new_partial)
            self.certainty.append(self.PREDICT_AND_IMG)


    def predict_visible_pixels(self, fish_region, last_pixels):
        """
         for collision between a paramecia and a fish, predict the pixels of the paramecia thar are not hidden
         by the fish.
         :param fish_region: fish's pixels
         :param last_pixels: paramecia's pixels from a frame before
         :return: array of tuples represent the pixels that belong to the visible paramecia
         """

        if len(last_pixels) == 0:
            return last_pixels
        dist = []
        for pixel in last_pixels:
            cur_dist = euclidean_distance(fish_region.coords, pixel)
            dist.append(min(cur_dist))

        dist_threshold = min(dist)+0.1
        new_partial = last_pixels[np.array(dist) > dist_threshold]
        while len(new_partial) > 0.9*(len(last_pixels)):
            new_partial = last_pixels[np.array(dist) > dist_threshold]
            dist_threshold += 0.1

        return new_partial


    def find_match_collision(self, before):
        """
        self is paramecia that probably after collision with the fish, and this function finds its para object
        that disappeared before the collision.
        # todo this function should be improved
        :param before: list of candidates paramecium, which disappeared because of collision with the fish
        and one of them is probably this 'new' paramecia.
        :return:
        """
        before_location = np.array([para.lastcoord for para in before])
        dist = euclidean_distance(self.lastcoord, before_location)

        if min(dist) < self.double_thresh:
            return np.array(before)[dist == min(dist)][0]
        else:
            return None

    def union(self, other, fish):
        """
        this union happens only to union paramecia before and after collision with the fish.
        this function not only unit the before and after paramecium, but also predict the visible pixels at each
        frame when only some part of the paramecia is visible and the rest is hidden by the fish.
        :param other:
        :param fish_coords:
        :return:
        """
        # todo maybe update more accurate with the new data: the filled data until new_coord and new_reg is only
        #  based on prediction from the data before, but not with the new data.
        new_coord = other.lastcoord
        new_reg = other.region_list[-1]

        self.update(new_coord, new_reg, self.FROM_IMG)
        i = -2
        while len(self.partial_pixel_list[i]) == 0:
            new_partial = self.predict_visible_pixels(fish, self.partial_pixel_list[i+1])
            self.partial_pixel_list[i] = new_partial
            i -= 1
            if len(new_partial) == 0:
                break


    def update(self, new_coord, new_reg, certainty):
        """
         update data about the current frame
         :param new_coord: tuple of double represent the center point
         :param new_reg: region object represents topographic information about the paramecia
         :param certainty: how the location determined
         :return:
         """

        self.location.append(new_coord)
        self.region_list.append(new_reg)
        self.all_pixel_list.append(new_reg.coords.astype(np.float))
        self.partial_pixel_list.append(new_reg.coords)
        self.certainty.append(certainty)

        self.lastcoord = new_coord

        self.waitindex = 0
        self.completedstmp = -1
        self.completed = False


class ParaTracker():
    def __init__(self, start_ind, end_ind, directory, pcw):
        self.directory = directory
        self.all_xy = []
        self.lifetime_thresh = 30
        self.long_xy = []
        self.after_fish_param_collision = []
        self.params_list = []
        self.completed_params = []
        self.collide_w_fish = []
        self.threshold_param_size = 10


    def single_frame(self, bin_event, frame, frame_index):
        params = find_param(frame)
        cur_frames_params = [para for para in params if para.area >= self.threshold_param_size]
        # parafilter_top is a list of tuple represents the location of each paramecia in a single frame

        self.after_fish_param_collision.append([])

        if frame_index == 0:
            self.params_list = [Para(frame_index, para.centroid, para) for para in cur_frames_params]

        # params_list is a list of para objects. asks if any elements of contour list are nearby each para.
        else:
            double_params_list = []
            for para in self.params_list:
                para.nearby(cur_frames_params, self.completed_params, frame_index, double_params_list)

            self.params_list += double_params_list
            new_params = [Para(frame_index, para.centroid, para) for para in cur_frames_params]
            not_new = []
            # new- is it inside the fish?
            # is it after two param
            # is it after param-fish

            fish = find_fish(frame)

            for para in new_params:
                collision_point = para.fish_collision(fish)
                #this paramecia is probably after collision with the fish
                if collision_point is not None:
                    self.after_fish_param_collision[-1].append(collision_point)
                    if len(self.collide_w_fish) > 0:
                        match = para.find_match_collision(self.collide_w_fish)
                        if match is not None:
                            match.union(para, fish)
                            not_new.append(para)
                            self.collide_w_fish.remove(match)
                            self.params_list.append(match)

            for para in not_new:
                new_params.remove(para)

            old_params = [para for para in self.completed_params if not para.completed]
            self.params_list = self.params_list + new_params + old_params
            new_complete = [para for para in self.params_list if para.completed]
            self.completed_params = self.completed_params + new_complete
            self.params_list = [para for para in self.params_list if not para.completed]
            self.completed_params = [para for para in self.completed_params if para.completed]
            # current para list params_list is cleaned of records that are complete.

            # saving params that collide with the fish
            for para in self.completed_params:
                if para.completedstmp < frame_index - para.waitmax or len(para.location) < self.lifetime_thresh:
                    continue
                fish = find_fish(bin_event[para.completedstmp - 1])

                collision_point = para.fish_collision(fish)
                if collision_point is not None:
                    # this condition is to check if the paramecia is really close to the fish and then we
                    # conclude it is a collision case
                    self.collide_w_fish.append(para)
                    para.create_prediction()

            for para in self.collide_w_fish:
                if para in self.completed_params:
                    self.completed_params.remove(para)
                fish = find_fish(bin_event[para.completedstmp - 1])
                para.predict_update(frame_index, fish)

    def find_paramecia(self, bin_event):
        for frame_index, frame in enumerate(bin_event):
            if frame_index % 100 == 0:
                print(frame_index)

            self.single_frame(bin_event, frame, frame_index)

        # post-process
        for para in self.collide_w_fish:
            para.location = para.location[:para.completedstmp]
            para.region_list = para.region_list[:para.completedstmp]
            para.all_pixel_list = para.all_pixel_list[:para.completedstmp]
            para.partial_pixel_list = para.partial_pixel_list[:para.completedstmp]
            para.certainty = para.partial_pixel_list[:para.completedstmp]

        all_xy = self.completed_params + self.params_list + self.collide_w_fish
        all_xy = sorted(all_xy, key=lambda x: len(x.location))
        all_xy.reverse()
        self.all_xy = [para for para in all_xy if len(para.location) > 0]
        self.long_xy = [para for para in self.all_xy if len(para.location) >= self.lifetime_thresh]

        print("num of paramc:", len(self.all_xy), " after filter:", len(self.long_xy))



def create_params_data(num, event_path, bin_event_path, noise_frame_path, binary=True):
    print("- - - - - - make binary " + num + " - - - - - - -")
    orig_event = np.fromfile(event_path, dtype=np.uint8)
    orig_event = np.reshape(orig_event, [orig_event.size // (FRAME_ROWS * FRAME_COLS), FRAME_COLS, FRAME_ROWS])
    noise_frame = np.load(noise_frame_path)

    if binary:
        bin_event = make_bin_event(orig_event, noise_frame)
        np.save(bin_event_path, bin_event)
    else:
        bin_event = np.load(bin_event_path)

    print("number of frames:", len(bin_event))

    print("- - - - - - mark paramecium " + num + " - - - - - - -")
    paraTracker = ParaTracker(0, len(bin_event), 0, 0)
    paraTracker.find_paramecia(bin_event)


    return paraTracker.all_xy, paraTracker.long_xy


def main_para():
    noise_frame_path = 'noise_frame.npy'
    event_num = '4'
    data_path = "..\\"

    event_path = data_path + "raw_data\\20200720-f3-"+event_num+".raw"
    bin_event_path = data_path + "output_np\\binary_events\\f3\\20200720-f3-" + event_num + ".npy"

    output_path = data_path + "output_movie\\f3\\20200720-f3-"+event_num+".avi"
    orig_event = np.fromfile(event_path, dtype=np.uint8)
    orig_event = np.reshape(orig_event, [orig_event.size // (FRAME_ROWS * FRAME_COLS), FRAME_COLS, FRAME_ROWS])
    params_data, long_params_data = create_params_data(event_num, event_path, bin_event_path, noise_frame_path, binary=True)
    from_params_data_to_movie(params_data, orig_event, output_path)

if __name__ == '__main__':
    main_para()


