import numpy as np
import pickle_To_p0
import itertools, sys
import cv2 as cv

# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def _create_opencv_tracker(tracker_type):
    tracker_types = {
    'MIL': cv.TrackerMIL_create,
    }
    assert tracker_type in tracker_types
    return tracker_types[tracker_type]()



class Tracker():



    newid = itertools.count()


    #initialize the tracker list
    def __init__(self, starting_point, state = True):
        self.point = starting_point
        self.id = Tracker.newid #make the same tracker, keep id
        self.state = state #if state is false, don't track
        self.dist_t = 2 #which threshold for closest bee?


    #update the tracker list for each frame, create/discard trackers
    '''in lucaskanade i have every 100 frames... keep that'''
    def update(self, new_frame, old_gray, frame_gray):

        lk_params = dict(   winSize  = (15,15),
                            maxLevel = 2,
                            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, np.array([self.point]), None, **lk_params)
        self.update_trackers(new_frame)

        # Pass the frame to each of the trackers
        for t in self.trackers:
            t.update(frame, old_gray, frame_gray)

        if st == 0: #if st is 0, don't track it, basically clean up anything that is false
            self.state = False
        else:
            self.point = p1[0]
        '''same thing as self.trackers = list(filter(lambda t: t.state, self._trackers))'''


    def update_point(self, point): #draw the line?
        self.point = point


#another function to give new xy points and reset, use p0 as new point


    def distance(self, bee_loc, tracker_point):
        return np.sqrt(np.square(bee_loc) + np.square(tracker_point))


    def create_tracker(self, p0):
        return Tracker(p0, True)



    def pair_trackers_with_detections(self, trackers, detections, dist, frame):
        """pairs trackers with smallest distance-- the matrix?"""
        unpaired_tracker_indices = list(range(len(trackers)))
        unpaired_detection_indices = list(range(len(detections)))

        dist = np.zeros((len(trackers), len(detections)))

        #create the matrix
        if frame % 100 == 0:
            for i, t in enumerate(trackers):
                for detection in detections:
                        dist.append(self.distance(pickle_To_p0.new_p0(), detection))


        # Threshold scores according to the provided threshold-- filtering
        for x in dist:
            if x >= dist_t:
                dist.remove(x)


        # So long as are there are distances > 0, keep pairing trackers with detections
        pairs = []
        while np.any(dist):
          # Find the smallest distance
            t, d = np.unravel_index(np.argmin(dist), dist.shape)
            pairs.append((trackers[t], detections[d]))
            # Mark both as having been paired
            unpaired_tracker_indices.remove(t)
            unpaired_detection_indices.remove(d)
            # Zero out any other scores involving the paired tracker and detection
            dist[t, :] = 0
            dist[:, d] = 0

        # Create a list of unpaired trackers and a list of unpaired detections
        unpaired_trackers = [trackers[i] for i in unpaired_tracker_indices]
        unpaired_detections = [detections[i] for i in unpaired_detection_indices]

        return pairs, unpaired_trackers, unpaired_detections

def make_Tracker(p0, cap):
    #p0 #is this for the matrices counting

    ret,frame = cap.read()
    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    trackers = []
    for i in p0:
        trackers.append(Tracker(p0))

    ret, frame = cap.read()

    for tr in trackers:
        tr.update(frame, old_gray, frame_gray)

    #create the different trackers
    td_pairs, unpaired_trackers, unpaired_detections = self.pair_trackers_with_detections(self.trackers, detections, self.dist_t, frame)

    #1. pairs of trackers and data -> override the tracker.point with the datas point
    trackers = [self.update_point()]

    #unpaired trackers --> nothing we can do about this
    trackers += unpaired_trackers

    #unparted data --> create new trackers
    trackers += [self.create_tracker(frame, detection) for detection in unpaired_detections]

    self.trackers = trackers
