import cv2 as cv
import itertools, sys
import numpy as np
import pickle_To_p0

class Tracker():

    newid = itertools.count()
    def __init__(self, x_pos, y_pos):
        self.x = x_pos
        self.y = y_pos
        self.id = Tracker.newid #make the same tracker, keep id

    def overwrite_location(self, x,y):
        self.x = x
        self.y = y

    #updates the x, y locations of the trackers
    # Only uses the optical flow to update location when there is no new detection
    def update(self, prev_frame, new_frame):
        lk_params = dict(   winSize  = (15,15),
                            maxLevel = 2,
                            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        #p1 returns new point

        p1, st, err = cv.calcOpticalFlowPyrLK(prev_frame, new_frame, np.array([[[np.float32(int(self.x)), np.float32(int(self.y))]]]), None, **lk_params)

        self.x = p1[0][0][0]
        self.y = p1[0][0][1]
        color1 = (list(np.random.choice(range(256), size=3)))
        color =[int(color1[0]), int(color1[1]), int(color1[2])]
        cv.circle(new_frame,(self.x, self.y), 10, color, -1)


    # def compute distance (list of detections)
    # return a list of distances betwenn this tracker and all the detections
    # detection is manually generated
    def compute_distance(self, detection):
        dist_from_one_t = []
        for x in range(len(detection)):
            dist_from_one_t.append(np.sqrt(np.square(detection[x][0][0] - self.x) + np.square(detection[x][0][1] - self.y)))
        print(dist_from_one_t)

        return dist_from_one_t

        #fix detection: detectio [1][0][0], list of distances
