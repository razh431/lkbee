import numpy as np
import cv2
import pickle
import math
import itertools
from matplotlib import pyplot as plt

class BeeTracker:
    """This class ... """

    # Initialize weights (of cost function) as class variables
    w_in = 0.4
    w_dist = 0.4
    w_or = 0.1
    w_or_in = 0.1
    newid = itertools.count()
    # list of 10 categorical colors from colorBrewer
    RGBs = [(166, 206, 227), (31, 120, 180), (178, 223, 138), (51, 160, 44), (251, 154, 153),
            (227, 26, 28), (253, 191, 111), (255, 127, 0), (202, 178, 214), (106, 61, 154)]
    color_list = [(np.divide(val, 255)) for val in RGBs]

    def __init__(self, frame_num, detection, axis):
        """detection is a tuple (xh, yh, xt, yt)"""
        # """detection is a tuple of tuples ((xh, yh),(xt, yt))"""

        self.id = next(BeeTracker.newid)
        self.frame_list = [frame_num]
        self.head_list = [(detection[0], detection[1])]  # [detection[0]]
        self.tail_list = [(detection[2], detection[3])]  # [detection[1]]
        self.centroid_list = [((detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2)]
        self.orientation_list = [self.calculate_orientation()]
        self.plot_color = self.color_list[self.id % len(self.color_list)]  # assign a color to tracker obj from color_list

        # Plot trajectory
        self.track_line, = axis.plot(self.centroid_list[0][0], self.centroid_list[0][1], color=self.plot_color,
                                     label=str(self.id))
        # Plot current centroid
        self.centroid_plot, = axis.plot(self.centroid[0], self.centroid[1], '.', color=self.plot_color, label=str(self.id))
        # Label current centroid position with string of id
        #
        # self.text = axis.text(self.centroid_list[0][0], self.centroid_list[0][1] + 1, str(self.id))
        # # Plot bee vector
        # self.vector_plot, = axis.plot((self.head[0], self.tail[0]), (self.head[1], self.tail[1]), 'k',
        #                               color='gray', label=str(self.id))
        # # Plot head
        # self.head_plot, = axis.plot(self.head[0], self.head[1], 'o', color=self.plot_color, label=str(self.id))

    @property
    def head(self):
        """Returns tuple of most recent location of bee head (x,y)"""
        return self.head_list[-1]

    @property
    def tail(self):
        """Returns tuple of most recent location of bee tail (x,y)"""
        return self.tail_list[-1]

    @property
    def dx(self):
        """Returns length of bee in the x direction"""
        return self.head[0] - self.tail[0]

    @property
    def dy(self):
        """Returns length of bee in the y direction"""
        return self.head[1] - self.tail[1]

    @property
    def vector(self):
        """Returns normal vector pointing in direction of bee"""
        return self.dx / math.sqrt(self.dx ** 2 + self.dy ** 2), self.dy / math.sqrt(self.dx ** 2 + self.dy ** 2)

    @property
    def recent_displacement(self):
        """Compute distance vector between current centroid and previous centroid"""
        return (self.centroid_list[-1][0] - self.centroid_list[-2][0],
                self.centroid_list[-1][1] - self.centroid_list[-2][1])

    @property
    def orientation(self):
        """Returns most recent orientation of bee"""
        return self.orientation_list[-1]

    @property
    def centroid(self):
        """Returns most recent centroid of bee"""
        return self.centroid_list[-1]

    @property
    def bee_length(self):
        """Returns length of bee"""
        return math.sqrt((self.head[0] - self.tail[0]) ** 2 + (self.head[1] - self.tail[1]) ** 2)

    @property
    def frame(self):
        """Returns video frame number associated with bee data"""
        return self.frame_list[-1]

    @property
    def time_since_detection(self):
        """Returns time since last pairing with detection was paired with tracker"""
        count = 0
        for i in len(self.head_list):
            if self.head_list[-(i + 1)] == 0:
                count = count + 1
            else:
                break

        return count

    @property
    def to_be_paired(self):
        """Tracker should not be paired with detections if the tracker has failed"""
        return self.time_since_detection > 0

    def calculate_orientation(self):
        """ Calculate orientation of vector from tail (moved to origin) to head """
        # Move vector to origin (x = x_head - x_tail, y = y_head - y_tail), then calculate orientation.
        x = self.head[0] - self.tail[0]
        y = self.head[1] - self.tail[1]
        return np.arctan2(y, x)

    def update_active_tracker_pos(self, frame_num, detection, axis):
        """Update tracker if there is a detection. Should only be called on Active Trackers that have been paired."""
        head = (detection[0], detection[1])
        tail = (detection[2], detection[3])
        self.frame_list.append(frame_num)
        self.centroid_list.append(((head[0] + tail[0]) / 2, (head[1] + tail[1]) / 2))
        self.head_list.append(head)
        self.tail_list.append(tail)
        self.orientation_list.append(self.calculate_orientation())

        # update tracker plot
        x = []
        y = []
        [x.append(pt[0]) for pt in self.centroid_list]
        [y.append(pt[1]) for pt in self.centroid_list]
        self.track_line.set_data(x, y)
        #axis.texts[self.id].set_x(self.centroid[0])
        #axis.texts[self.id].set_y(self.centroid[1]+2)
        self.centroid_plot.set_data(self.centroid[0], self.centroid[1])
        # self.head_plot.set_data(self.head[0], self.head[1])
        # self.vector_plot.set_data((self.head[0], self.tail[0]), (self.head[1], self.tail[1]))


    def update_inactive_tracker_pos(self, frame_num):
        self.frame_list.append(frame_num)
        self.centroid_list.append((0, 0))
        self.head_list.append((0, 0))
        self.tail_list.append((0, 0))
        self.orientation_list.append(0)
        # set plot to invisible
        self.show(False)

    def __str__(self):
        """ Modify string passed to print statement to show most recent data, rather than just object pointer """
        string = 'id = ' + str(int(self.id)) \
                 + ', frame = ' + str(self.frame) \
                 + ', head = ' + str((round(self.head[0], 1), round(self.head[1], 1))) \
                 + ', tail = ' + str((round(self.tail[0], 1), round(self.tail[1], 1))) \
                 + ', centroid = ' + str((round(self.centroid[0], 1), round(self.centroid[1], 1))) \
                 + ', orientation = ' + str(round(self.orientation, 1))
        return string

    def get_orientation_cost(self, detections):
        """
        detections (input): a list of detections for which each element is a tuple of length 4.
                        For example: (xh, yh, xt, yt)
        corr_list (return): a list of correlations between the objects vector and the detections' vectors
        """
        # Move each detection to origin
        detections_origin = [(xh - xt, yh - yt) for xh, yh, xt, yt, in detections]

        # Normalize each detection
        detections_norm = [(dx / math.sqrt(dx ** 2 + dy ** 2), dy / math.sqrt(dx ** 2 + dy ** 2)) for dx, dy in
                           detections_origin]

        # Take inner product of bee vector with detection vector
        return np.array([d_dx * self.vector[0] + d_dy * self.vector[1] for d_dx, d_dy in detections_norm])

    def get_centroid_distance_cost(self, detections):
        """
        detections (input): a list of detections for which each element is a tuple of length 4.
                        For example: (xh, yh, xt, yt)
        distances (return): a list of the distance from trackers centroid to each detections centroid
        """

        # detections_centroid = [((head[0] + tail[0])/2, (head[1] + tail[1])/2) for head, tail in detections]
        detections_centroid = [((xh + xt) / 2, (yh + yt) / 2) for xh, yh, xt, yt in detections]

        dist_cost = np.array([math.sqrt((d_x - self.centroid[0]) ** 2 +
                                        (d_y - self.centroid[1]) ** 2) for d_x, d_y in detections_centroid])

        dist_cost = np.array([100000000 if cost > 1.5*self.bee_length else cost for cost in dist_cost])

        return dist_cost

    def get_inertial_cost(self, detections):
        """
        detections (input): a list of detections for which each element is a tuple of length 4.
                For example: (xh, yh, xt, yt)
        inertial_cost (return): a list of distances from the projected position (based on last velocity)
         and the detection.
        """

        if len(self.head_list) == 1:
            return np.array([0] * len(detections))
        else:
            delta = self.recent_displacement  # displacement since previous timestep

            # compute projections from current location based on delta
            projection = (self.centroid[0] + delta[0], self.centroid[1] + delta[1])
            # compute centroids of detections
            detections_centroid = [((xh + xt) / 2, (yh + yt) / 2) for xh, yh, xt, yt in detections]

            inertial_cost = np.array([math.sqrt((d_x - projection[0]) ** 2 + (d_y - projection[1]) ** 2)
                                      for d_x, d_y in detections_centroid])

            # Make cost infinite if distance is greater than 3 bee lengths
            inertial_cost = np.array([100000000 if cost > 1.5*self.bee_length else cost for cost in inertial_cost])

        return inertial_cost

    def get_orientation_inertial_cost(self, detections):
        """
        detections (input): a list of detections for which each element is a tuple of length 4.
                For example: (xh, yh, xt, yt)
        inertial_cost (return): a list of distances from the projected position (based on last velocity)
         and the detection.
        """

        if len(self.head_list) == 1:
            return np.array([0] * len(detections))
        else:
            delta = self.recent_displacement  # displacement since previous timestep

            # compute
            projected_delta = (
                (math.sqrt(delta[0] ** 2 + delta[1] ** 2) * (self.head[0] - self.tail[0])) / self.bee_length,
                (math.sqrt(delta[0] ** 2 + delta[1] ** 2) * (self.head[1] - self.tail[1])) / self.bee_length)

            # compute projected position
            projection = (self.centroid[0] + projected_delta[0], self.centroid[1] + projected_delta[1])

            # compute centroids of detections
            detections_centroid = [((xh + xt) / 2, (yh + yt) / 2) for xh, yh, xt, yt in detections]

            or_in_cost =  np.array([math.sqrt((d_x - projection[0]) ** 2 + (d_y - projection[1]) ** 2)
                             for d_x, d_y in detections_centroid])

            # Make cost infinite if distance is greater than 3 bee lengths
            or_in_cost = np.array([100000000 if cost > 1.5*self.bee_length else cost for cost in or_in_cost])

            return or_in_cost

    def get_costs(self, detections):
        in_cost = self.get_inertial_cost(detections)  # Get euclidean distances between centroids
        dist_cost = self.get_centroid_distance_cost(detections)
        or_cost = self.get_orientation_cost(detections)
        or_in_cost = self.get_orientation_inertial_cost(detections)
        costs = self.w_in * in_cost + \
                self.w_dist * dist_cost + \
                self.w_or * or_cost + \
                self.w_or_in * or_in_cost
        return costs

    def show(self, to_show=False):
        """If to_show (bool) is True, the line will be visible. If to to_show is False, the line is not invisible """
        self.track_line.set_visible(to_show)
        # self.text.set_visible(to_show)
        self.centroid_plot.set_visible(to_show)
        # self.head_plot.set_visible(to_show)
        # self.vector_plot.set_visible(to_show)


class TrackerManager:
    """
    This class manages the bee trackers.
    """

    # Define class variables
    cost_threshold = 300  # threshold cost for pairing trackers with new detections

    def __init__(self):
        # trackers is a list of BeeTracker objects
        self.active_trackers = []
        self.inactive_trackers = []
        # set up figure
        self.fig, self.ax = plt.subplots()


    def generate_cost_matrix(self, detections):
        cost_mat = np.zeros((len(self.active_trackers), len(detections)))

        for i, t in enumerate(self.active_trackers):
            cost_mat[i, :] = t.get_costs(detections)
        # print('cost_matrix = ', cost_mat)
        return cost_mat

    def pair_trackers_with_detections(self, detections):

        unpaired_tracker_indices = list(range(len(self.active_trackers)))
        unpaired_detections_indices = list(range(len(detections)))

        indices = []
        for t_n in range(0, len(self.active_trackers)):
            for d_n in range(0, len(detections)):
                indices.append((t_n, d_n))

        # Calculate cost matrix
        cost_mat = self.generate_cost_matrix(detections)
        # Flatten cost matrix
        flattened_cost_list = cost_mat.flatten().tolist()
        # Sort cost and indices based on cost, store as list of tuples [(cost, (tracker_index, decttion_index)), ...]
        sorted_indices = [(c, i) for c, i in sorted(zip(flattened_cost_list, indices))]

        pairs = []
        for c, (t, d) in sorted_indices:
            if c < self.cost_threshold:
                if t in unpaired_tracker_indices and d in unpaired_detections_indices:
                    pairs.append((t, d))
                    unpaired_tracker_indices.remove(t)
                    unpaired_detections_indices.remove(d)

        return pairs, unpaired_tracker_indices, unpaired_detections_indices

    def update_trackers(self, im, frame_num, detections):

        self.ax.imshow(im)

        pairs_idx, unpaired_trackers_idx, unpaired_detections_idx = self.pair_trackers_with_detections(detections)

        # Update the positions of active paired trackers and add them to plot
        for t, d in pairs_idx:
            self.active_trackers[t].update_active_tracker_pos(frame_num, detections[d], self.ax)

        # Move unpaired trackers to inactive tracker list
        for t in unpaired_trackers_idx:
            self.inactive_trackers.append(self.active_trackers[t])
            # Tag active trackers to be removed
            self.active_trackers[t] = 0
        # Remove tagged trackers
        self.active_trackers = [ x for x in self.active_trackers if 0 != x ]

        # Update the position of inactive
        for it in self.inactive_trackers:
            it.update_inactive_tracker_pos(frame_num)

        # Create new trackers for unpaired detections
        for d in unpaired_detections_idx:
            self.active_trackers.append(BeeTracker(frame_num, detections[d], self.ax))

        plt.show()
        plt.pause(0.001)

# Video file
input_video_FID = '/Users/rachelzhou/Research/build_opencv/opencv/samples/python/rachelResearch/buzzbuzz.mp4'
input_video = cv2.VideoCapture(input_video_FID)
#r = [718, 640, 2084, 903]  # ROI (must match original)
r = [2171, 1050, 556, 528]

# Get input database
pickle_fid = '/Users/rachelzhou/Research/build_opencv/opencv/samples/python/rachelResearch/blob_detection_output_5g_2020_03_15_20_49_02.pickle'
# pickle_fid = '/Users/Jake/PycharmProjects/trackBees2/manual_digitizing/blob_detection_output_5g_2020_03_10_18_36_11.pickle'
# pickle_fid = '/Users/Jake/PycharmProjects/trackBees2/manual_digitizing/blob_detection_output_5G.pickle'
# pickle_fid = '/Users/Jake/PycharmProjects/trackBees2/manual_digitizing/rachel_5G_data.pickle'
with open(pickle_fid, 'rb') as f:
    data_dict = pickle.load(f)

# Get a list of all existing keys
frame_key_list = data_dict.keys()
frame_key_list_int = [int(i) for i in list(frame_key_list)]
frame_key_sorted = sorted(frame_key_list_int)  # integers

# Get first frame of data
i = 0
curr_frame = frame_key_sorted[0]
x_head, y_head, x_tail, y_tail = data_dict[str(curr_frame)]

# Initialize first detections
detections = []
# [detections.append((xh+r[0], yh+r[1], xt+r[0], yt+r[1])) for xh, yh, xt, yt in zip(x_head, y_head, x_tail, y_tail)]
[detections.append((xh, yh, xt, yt)) for xh, yh, xt, yt in zip(x_head, y_head, x_tail, y_tail)]

# Construct tracker manager object
manager = TrackerManager()

for i in range(1, len(frame_key_sorted)):

    # Get data from current frame
    curr_frame = frame_key_sorted[i]
    x_head, y_head, x_tail, y_tail = data_dict[str(curr_frame)]
    if not x_head:
        continue

    # Make detections list
    detections = []
    # [detections.append((xh+r[0], yh+r[1], xt+r[0], yt+r[1])) for xh, yh, xt, yt in zip(x_head, y_head, x_tail, y_tail)]
    [detections.append((xh, yh, xt, yt)) for xh, yh, xt, yt in zip(x_head, y_head, x_tail, y_tail)]

    # Generate cost matrix
    cost_matrix = manager.generate_cost_matrix(detections)

    # Get frame
    input_video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
    ret, im = input_video.read()
    im = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    # Update trackers and make new trackers from unpaired detections
    manager.update_trackers(im, curr_frame, detections)

    print('curr_frame = ', curr_frame)
