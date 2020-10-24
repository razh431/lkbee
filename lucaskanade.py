import pickle_To_p0
import example
import pickle

import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation.')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

#frames
cap = cv.VideoCapture(args.image) #the frame 3500
startFrame = 4500
cap.set(1, startFrame)

#pickle files, returns result in a dict
pickle_in = open("5G_state_data_copy_2.pickle", "rb")
result = pickle.load(pickle_in)

 ## params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 300,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(300,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

#stuff to find p0, which looks something like below
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# p0 = np.array([[[3680., 715.3647]], [[301.,104.]], [[272.,104.]], [[308.,210.]], [[264.,151.]], [[182.,186.]], [[175.,187.]], [[383.,211.]], [[305.,88.]], [[370.,93.]]], np.float32)

p0 = pickle_To_p0.new_p0(startFrame, result)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

print('here')
while(1):
    ret,frame = cap.read()

    #the current frame num
    frame_num = int(cap.get(cv.CAP_PROP_POS_FRAMES))

    #if the frame matches any that was manually generated, reset p0 and tracks
    if str(frame_num) in list(result.keys()):
        p0 = pickle_To_p0.new_p0(frame_num, result)
        mask = np.ones_like(frame)

        # example.make_Tracker(p0, cap)


#shouldn't be touching this part
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1] #st is 0 when can't track so list is <= origonal
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)


    img = cv.add(frame,mask)
    print(img)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)


#notes for rachel:
#p0 in right data
#only scenting, walking bees
#keep track of frame number--> read
#check if you reached a frame that has data
#keys --> returns an object, convert to list
#check which frame, if on they key of the list, grabbing the data, use as p0
#frame 99, calculate distances between 2 points

#
# #make p0 list
# for every index in p0, make a new tracker
#
# while true:
#     read frame
#     for t in trackers:
#         t.update(frame)
#
#
#
#
# in tracker class:
#
# def init:
#     x = x
#     y  = y
#
#
# def update(previous frame, new frame): #new
#     new_x, new_y = cv. optical flow (old frame, new frame, ___)
#     self.x = new_x
#     self.y = new_y
