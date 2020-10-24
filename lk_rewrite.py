import pickle
import pickle_To_p0
import numpy as np
import cv2 as cv
import argparse
import tracker
from tracker import Tracker

#ArgumentParser stuff
#
# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation.')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()
#
# #frames
cap = cv.VideoCapture("bees.mp4") #the frame 3500
startFrame = 5360
cap.set(1, startFrame)

#pickle files, returns result in a dict
pickle_in = open("rachel_5G_data.pickle", "rb")
result = pickle.load(pickle_in)

p0 = pickle_To_p0.new_p0(startFrame, result)
p0 = p0[0:5]

def lk_rewrite():
    #make p0 list
    tracker_list = []
    ret, prev_frame = cap.read()
    for index in p0:
        #p0 returns [[x, y]] for optical flow to work i think
        t = tracker.Tracker(index[0][0], index[0][1])
        tracker_list.append(t)

    while(1):
        cv.imshow('frame', cv.resize(prev_frame,(int(prev_frame.shape[1]/3),int(prev_frame.shape[0]/3))))
        cv.waitKey(1)

        ret, new_frame = cap.read()

        # distances[0][0] is distance between tracker 0 and detection 0
        distances = []
        # Get detection points

        #current frame in an int
        frame_num = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        print(frame_num)

        #if the current frame is one of the manually generated ones, then do detections
        if str(frame_num) in list(result.keys()):

            detections = pickle_To_p0.new_p0(frame_num, result)
            # detections = detections[0:5]

            # For each tracker, compute distance to each point
            for t in tracker_list:
                distances.append(t.compute_distance(detections))

            #pair them up
            pair_of_td, unpaired_trackers, unpaired_detections = pair_t_d(tracker_list, detections, distances)
            after_paired(pair_of_td, unpaired_trackers, unpaired_detections)

        for t in tracker_list:
           t.update(prev_frame, new_frame)

        after_paired([pair_of_td])
        prev_frame = new_frame


def pair_t_d(trackers, detections, distances):
    # unpaired_tracker_indices = list(range(len(trackers)))
    # unpaired_detection_indices = list(range(len(detections)))
    unpaired_tracker_indices = []
    unpaired_detection_indices = []
    dist = np.zeros((len(trackers), len(detections)))

    #create the matrix rows: trackers, detections: column
    for i,t in enumerate(trackers):
        for detection in detections:
            dist[i, :] = distances[i]

    pair_of_td = []

    while(len(np.where(dist > -1)[0]) > 0):
        #for every one that is paired, change it to -1
        b = np.ravel(dist)
        index = np.where(b >= 0, b, np.inf).argmin()
        tnum = index//len(dist) #row
        dnum = index%len(dist) #column

        if dist[tnum][dnum] < 30:
        # dist[i,:] = np.array([-1]*len(dist[i,:])])-1
            for x in range(len(dist[tnum])):
                dist[tnum][x] = -1
                dist[x][dnum] = -1
            pair_of_td.append([tnum]) #wouldn't detections and trackers match indexes
        else:
            unpaired_tracker_indices.append([tnum])
            unpaired_detection_indices.append([dnum])
            break '''fix this thing'''

    unpaired_trackers = [trackers[i] for i in unpaired_tracker_indices]
    unpaired_detections = [detections[i] for i in unpaired_detection_indices]
    #
    # print(dist)
    # print(pair_of_td)
    return pair_of_td, unpaired_trackers, unpaired_detections

def after_paired(pair_of_td, unpaired_trackers, unpaired_detections):
    for x in pair_of_td():
        overwrite_location() #get detecion x and y

    for index in unpaired_detections:

        t = tracker.Tracker(detections[0][0], [detection][0][1]) #get p0th element which contains the x and y
        tracker_list.append(t)


#         overwrite update for each paired, tracker, override location of tracker
#



        #for each pair, update xy location of tracker with the detection with each of the pairs
        #all of the detections that didnt get pairs, create new trackers, append to list of trackers
        #for all the objects that iddnt have detections, use lk, change state,
            #keep track of how many frames it's been since detections


lk_rewrite()





#distance, matrix, updated points-- rewrite?
