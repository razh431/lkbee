import pickle
import numpy as np

pickle_in = open("5G_state_data_copy_2.pickle", "rb")
result = pickle.load(pickle_in)

#return list of p0
def new_p0(startFrame, result):

    """change pickle file"""
    key = str(startFrame)
    x_head, y_head, x_tail, y_tail = result[key]

    x_head1 = []
    y_head1 = []
    # for x in range(len(state)):
    #     if state[x] == 'scenting' or state[x] == 'walking':
    #
    #         x_head1.append(np.float32(x_head[x]))
    #         y_head1.append(np.float32(y_head[x]))
    for x in range(len(x_head)):
        x_head1.append(np.float32(x_head[x]))
        y_head1.append(np.float32(y_head[x]))

    xy_coord = [list(xy) for xy in zip(x_head1, y_head1)]

    p0 = np.array(xy_coord, np.float32)
    p0 = np.expand_dims(xy_coord, 1)

    return p0
#
# #return a single p0 point
# def single_point(startFrame, point_num):

def tries():
    objects = []
    with (open("5G_state_data_copy_2.pickle", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    print(objects)

# new_p0(5360, result)
tries()
