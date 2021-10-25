
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

def perfect_match_equel(list_a, list_b):
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


def perfect_match(list_a, list_b):
    perfect = {}
    if len(list_a) == len(list_b):
        best_permute_a = list_a
        score, best_permute_b = perfect_match_equel(list_a, list_b)

    else:
        if len(list_a) < len(list_b):
            len_a = len(list_a)
            best_permute_a = list_a
            best_permute_b = list_b[:len_a]
            min_score = 100
            for i in range(len(list_b)-len(list_a)):
                permute_b = list_b[i:len_a+i]
                score, permute_b = perfect_match_equel(list_a, permute_b)
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
                score, permute_a = perfect_match_equel(list_b, permute_a)
                if score < min_score:
                    min_score = score
                    best_permute_a = permute_a

    for a, b in zip(best_permute_a, best_permute_b):
        perfect[a] = b
    return perfect


class Para:
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
        self.direction = []
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

        self.double_thresh = 10
        self.double = False
        self.double_smp = -1
        self.num_double = 1

        # the delta of frames to calculate the direction vector
        self.direction_thresh = 1
        self.region_list = [region]
        self.all_pixel_list = [region.coords.astype(np.float)]
        self.partial_pixel_list = [region.coords]
        self.certainty = [self.FROM_IMG]
        self.predict_x = None
        self.predict_y = None
        self.predict_x_direction = None
        self.predict_y_direction = None

    def endrecord(self, timestmp):
        # CHOPS OFF THE HANGING LEN(WAITMAX) PORTION WHERE IT COULD'NT FIND PARTNER
        self.location = self.location[:-self.waitmax]
        self.direction = self.direction[:-self.waitmax]
        self.region_list = self.region_list[:-self.waitmax]
        self.all_pixel_list = self.all_pixel_list[:-self.waitmax]
        self.partial_pixel_list = self.partial_pixel_list[:-self.waitmax]
        self.certainty = self.certainty[:-self.waitmax]
        self.completed = True
        self.completedstmp = timestmp-self.waitmax

    def nearby(self, cont_list, completed_list, timestmp, double_options, maybe_double):

        # first, search the list of contours found in the image to see if any of them are near
        # the previous position of this Para.
        loc_list = [para.centroid for para in cont_list]

        if len(cont_list) == 0:
            pcoords = np.array([])
            pcoords_para = np.array([])
        else:
            loc_arr = np.array(loc_list)
            para_arr = np.array(cont_list)
            dist = euclidean_distance(loc_arr, np.array(self.lastcoord))

            pcoords = loc_arr[dist < self.thresh]
            pcoords_para = para_arr[dist < self.thresh]

        # if there's nothing found, add 1 to the waitindex, and say current position is the last position
        if pcoords.shape[0] == 0:

            self.location.append(self.lastcoord)
            self.region_list.append(self.region_list[-1])
            self.all_pixel_list.append(self.all_pixel_list[-1])
            self.partial_pixel_list.append(self.all_pixel_list[-1])
            self.certainty.append(self.REPEAT_LAST)
            if len(self.location) > 1:
                self.direction.append(np.array([0, 0]))
            if self.waitindex == self.waitmax:
                # end record if you've gone 'waitmax' frames without finding anything.
                # this value greatly changes things. its a delicate balance between losing the para and waiting too
                # long while another para enters
                self.endrecord(timestmp)
            self.waitindex += 1

        # this case is only one contour is within the threshold distance to this Para.
        elif pcoords.shape[0] == 1:

            newcoord = pcoords[0]
            newreg = pcoords_para[0]
            self.update(newcoord, newreg, self.FROM_IMG)
            cont_list.remove(newreg)

        # this case is that two or more contours fit threshold distance.
        elif pcoords.shape[0] > 1:
            # self.double = True
            # print('double: where, ', self.lastcoord, 'when, ', self.timestamp+len(self.location))
            if self not in maybe_double:
                maybe_double.append(self)
            # create list of close 'doubles'

            after_potential = pcoords_para
            before_potential = self.find_close_before_double(completed_list, timestmp)
            para_reg_match = self.match_double(before_potential, after_potential, timestmp)
            for para in para_reg_match:
                completed_list.remove(para)
                double_options.append(para)
                cont_list.remove(para_reg_match[para])

        return cont_list

    def match_double(self, before, after, cur_frame):
        predicted_location = []
        for para in before+[self]:
            para.create_prediction()
            x = para.predict_x(cur_frame)
            y = para.predict_y(cur_frame)
            predicted_location.append((x, y))

        real_location = [reg.centroid for reg in after]
        perfect = perfect_match(predicted_location, real_location)
        para_reg_match = {}
        for predict in perfect:
            para_index = predicted_location.index(predict)
            para = (before+[self])[para_index]
            real_coord = perfect[predict]
            real_index = real_location.index(real_coord)
            real_reg = after[real_index]

            if para.completed:
                para_reg_match[para] = real_reg
                para.location += self.location[para.completedstmp:]
                para.region_list += self.region_list[para.completedstmp:]
                para.all_pixel_list += self.all_pixel_list[para.completedstmp:]
                para.certainty += [self.DOUBLE_PARA]*(cur_frame-para.completedstmp)
            para.update(real_coord, real_reg, self.FROM_IMG)
        return para_reg_match

    def find_close_before_double(self, completed_list, cur_frame):
        close_before_double = []
        for para in completed_list:
            frame_before = cur_frame-para.completedstmp
            if frame_before > len(self.location):
                continue
            dist = euclidean_distance(np.array([para.lastcoord]), self.location[-frame_before])
            if dist[0] < self.double_thresh:
                close_before_double.append(para)

        return close_before_double

    def param_trajectory(self):
        x = [loc[1] for loc in self.location]
        y = [loc[0] for loc in self.location]

        fig, ax = plt.subplots()
        ax.plot(x, y, '--g', linewidth=0.2)
        plt.tight_layout()
        plt.show()


    def fish_collision(self, fish):

        dist = euclidean_distance(fish.coords, np.array(self.lastcoord))
        if min(dist) < 10:
            return fish.coords[dist == min(dist)][0]
        if min(dist) < self.thresh+2:
            return fish.coords[dist == min(dist)][0]
        else:
            return None

    def create_prediction(self):
        fitt_from = (len(self.location) // 3) * 2
        predict_x, predict_y, x_direction, y_direction = predict2d(np.arange(fitt_from, len(self.location)), self.location[fitt_from:])
        self.predict_x = predict_x
        self.predict_y = predict_y
        self.predict_x_direction = x_direction
        self.predict_y_direction = y_direction

    def predict_update(self, go_to_frame, fish):
        for i in range(len(self.location), go_to_frame):
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
        before_location = np.array([para.lastcoord for para in before])
        dist = euclidean_distance(self.lastcoord, before_location)

        if min(dist) < 10:
            return np.array(before)[dist == min(dist)][0]
        else:
            return None

    def find_why_completed(self, fish, params_list):
        params = np.array([para.lastcoord for para in params_list])

        dist_fish = euclidean_distance(fish.coords, np.array(self.lastcoord))
        min_fish = min(dist_fish)
        if len(params_list) == 0:
            if min_fish < 10:
                return Para.FISH
            else:
                return Para.ELSE

        else:
            dist_params = euclidean_distance(self.lastcoord, params)
            min_params = min(dist_params)

            if min_fish < min_params:
                if min_fish < 10:
                    return Para.FISH
                else:
                    return Para.ELSE

            else:
                if min_params < 10:
                    return Para.PARA
                else:
                    return Para.ELSE

    def union(self, other, fish):
        #todo maybe update more accurate with the new data
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

        self.location.append(new_coord)
        self.region_list.append(new_reg)
        self.all_pixel_list.append(new_reg.coords.astype(np.float))
        self.partial_pixel_list.append(new_reg.coords)
        self.certainty.append(certainty)
        if len(self.location) > 1:
            self.direction.append(np.array(new_coord) - self.lastcoord)
        self.lastcoord = new_coord

        self.waitindex = 0
        self.completedstmp = -1
        self.completed = False





class ParaMaster():
    def __init__(self, start_ind, end_ind, directory, pcw):
        self.pcw = pcw
        self.directory = directory
        self.all_xy = []
        self.xyzrecords = []
        self.framewindow = [start_ind, end_ind]
        self.para3Dcoords = []
        self.distance_thresh = 100
        self.length_thresh = 30
        self.time_thresh = 60
        self.filter_width = 5
        self.paravectors = []
        self.paravectors_normalized = []
        self.dots = []
        self.makemovies = True
        self.topframes = deque()
        self.topframes_original = deque()
        self.velocity_mags = []
        self.length_map = np.vectorize(lambda a: a.shape[0])
        self.long_xy = []
        self.short_xy = []
        self.before_fish_param_collision = []
        self.after_fish_param_collision = []
        self.maybe_double = []
        self.p_t = []
        self.completed_t = []
        self.collide_w_fish = []
        self.threshold_param_size = 10


    def single_frame(self, bin_event, frame, frame_index):
        params = find_param(frame)
        cur_frames_params = [para for para in params if para.area >= self.threshold_param_size]
        # parafilter_top is a list of tuple represents the location of each paramecia in a single frame

        self.before_fish_param_collision.append([])
        self.after_fish_param_collision.append([])

        if frame_index == 0:
            self.p_t = [Para(frame_index, para.centroid, para) for para in cur_frames_params]

        # p_t is a list of para objects. asks if any elements of contour list are nearby each para p.
        else:

            double = []
            for para in self.p_t:

                para.nearby(cur_frames_params, self.completed_t, frame_index, double, self.maybe_double)

            self.p_t += double
            newpara_t = [Para(frame_index, para.centroid, para) for para in cur_frames_params]
            not_new = []
            # new- is it inside the fish?
            # is it after two param
            # is it after param-fish

            fish = find_fish(frame)  # todo take from sapir's fish

            for para in newpara_t:
                collision_point = para.fish_collision(fish)
                if collision_point is not None:
                    self.after_fish_param_collision[-1].append(collision_point)
                    if len(self.collide_w_fish) > 0:
                        match = para.find_match_collision(self.collide_w_fish)
                        if match is not None:
                            match.union(para, fish)
                            not_new.append(para)
                            self.collide_w_fish.remove(match)
                            self.p_t.append(match)

            for para in not_new:
                newpara_t.remove(para)

            old_para = [para for para in self.completed_t if not para.completed]
            self.p_t = self.p_t + newpara_t + old_para
            new_complete = [para for para in self.p_t if para.completed]
            self.completed_t = self.completed_t + new_complete
            self.p_t = [para for para in self.p_t if not para.completed]
            self.completed_t = [para for para in self.completed_t if para.completed]
            # current para list p_t is cleaned of records that are complete.

            # saving params that collide with the fish
            for para in self.completed_t:
                if para.completedstmp < frame_index - para.waitmax or len(para.location) < self.length_thresh:
                    continue
                fish = find_fish(
                    bin_event[para.completedstmp - 1])  # todo take sapir's contour to check if this is an eye

                collision_point = para.fish_collision(fish)
                if collision_point is not None:
                    # this condition is to check if the paramecia is really close to the fish and then we
                    # conclude it is a collision case

                    self.before_fish_param_collision[-1].append(collision_point)
                    self.collide_w_fish.append(para)
                    para.create_prediction()

            for para in self.collide_w_fish:

                if para in self.completed_t:
                    self.completed_t.remove(para)
                fish = find_fish(
                    bin_event[para.completedstmp - 1])  # todo take sapir's contour to check if this is an eye
                para.predict_update(frame_index, fish)

    def findpara(self, bin_event):
        for frame_index, frame in enumerate(bin_event):
            if frame_index % 100 == 0:
                print(frame_index)

            self.single_frame(bin_event, frame, frame_index)

        # post-process
        for para in self.collide_w_fish:

            para.location = para.location[:para.completedstmp]
            para.region_list = para.region_list[:para.completedstmp]
            para.all_pixel_list = para.all_pixel_list[:para.completedstmp]
            para.partial_pixel_list = para.partial_pixel_list[:para.completedstmp]#todo maybe we need this information
            para.certainty = para.partial_pixel_list[:para.completedstmp]
        all_xy = self.completed_t + self.p_t + self.collide_w_fish
        all_xy = sorted(all_xy, key=lambda x: len(x.location))
        all_xy.reverse()
        self.all_xy = [para for para in all_xy if len(para.location) > 0]

        self.long_xy = [para for para in self.all_xy if len(para.location) >= self.length_thresh]

        self.short_xy = [para for para in self.all_xy if len(para.location) < self.length_thresh]

        print("num of paramc: ", len(self.all_xy), " after: ", len(self.long_xy))


# def generate_10_movies():
#     noise_frame_path = 'noise_frame.npy'
#     double_list = []
#
#     for i in range(1, 11):
#         num = str(i)
#         event_path = 'Z:\Lab-Shared\Data\FeedingAssay2020\\20200720-f3\\20200720-f3-' + num + '.raw'
#         bin_event_path = '..\output_np\\binary_events\\20200720-f3-' + num + '.npy'
#         output_path = 'Z:\yuval.pundakmint\extractData\\20200720-f3\\20200720-f3-' + num + '.avi'
#         generate_movie(event_path, bin_event_path, output_path, noise_frame_path, binary=False)
#         double_list.append(Para.doubles)
#
#     print(double_list)


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

    print(len(bin_event))

    print("- - - - - - mark paramecium " + num + "- - - - - - -")
    paraMaster = ParaMaster(0, len(bin_event), 0, 0)
    paraMaster.findpara(bin_event)
    for para in paraMaster.all_xy:
        if para.id == 7:
            para_7 = para
            break

    return paraMaster.all_xy, paraMaster.long_xy, paraMaster.short_xy, para_7


def main_para():
    noise_frame_path = 'noise_frame.npy'
    event_num = '1'
    data_path = "..\\"

    event_path = data_path + "raw_data\\20200720-f3-"+event_num+".raw"
    bin_event_path = data_path + "output_np\\binary_events\\f3\\20200720-f3-" + event_num + ".npy"

    output_path = data_path + "output_movie\\f3\\20200720-f3-"+event_num+".avi"
    orig_event = np.fromfile(event_path, dtype=np.uint8)
    orig_event = np.reshape(orig_event, [orig_event.size // (FRAME_ROWS * FRAME_COLS), FRAME_COLS, FRAME_ROWS])
    params_data, long_params_data, short_params_data, para_7 = create_params_data(event_num, event_path, bin_event_path, noise_frame_path, binary=True)
    from_params_data_to_movie(params_data, orig_event, output_path)

if __name__ == '__main__':
    main_para()


