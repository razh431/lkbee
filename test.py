import numpy as np

def rachel1():
    result = {
        '3500': [[0.111, 0.222, 0.333], [0.444, 0.555, 0.666], ['crying', 'laughing', 'screaming']],
        '4000': [[1.175, 1.378, 1.333], [1.111, 1.222, 1.444], ['crying', 'laughing', 'crying']],
        '5000': [[2.000, 2.111, 2.222], [2.333, 2.444, 2.555], ['crying', 'laughing', 'crying']],
    }

    key = '3500'


    x_head, y_head, state = result[key]

    #print the ones that are crying and screaming
    for x in range(len(state)):
        if state[x] != 'crying' and state[x] != 'screaming':
            x_head[x] = -1
            y_head[x] = -1

    x_head = [value for value in x_head if value != -1]
    y_head = [value for value in y_head if value != -1]
    state = [value for value in state if (value == 'crying' or value == 'screaming')]

    print(x_head)
    print(y_head)
    print(state)

    # q = {}
    # for x in range(len(result[key][0])):
    #     val = [[result[key][0][x], result[key][1][x]]] #i did 0 and 1 for head
    #     if q != {}:
    #         q[key] += [val]
    #     else:
    #         q[key] = [val]

    #
    x_head = np.array(x_head, np.float32)
    y_head = np.array(y_head, np.float32)
    p0 = list(zip(x_head, y_head))

    print(p0)


#
# #     p = {}
# #
# #     p["3500"] = [[result['3500'][0][0], result['3500'][1][0]],             [result['3500'][0][1], result['3500'][1][1]],[result['3500'][0][2], result['3500'][1][2]]]
# #
# #     # print(p)
# #
# #     #want something like [[0.111, 0.444], [0.222, 0.555], [0.333, 0.666]]
# #
# #     q = {}
# #
# #     for x in range(len(result['3500'][0])):
# #         val = [[result['3500'][0][x], result['3500'][1][x]]]
# #         if q != {}:
# #             q['3500'] += [val]
# #         else:
# #             q['3500'] = [val]
# #
# #     # print(q)
# # #
# # # [[[[0.118 0.111]
# # #    [0.123 0.222]]]]
# #
# #
# #     # for key,val in result.items():
# #     #     print(len(result[key]))
# #     #
# #     #     p[key] = [val[0], val[1]]
# #
# #     a = np.array(list(q.values()), np.float32)[0] #idk why there's an extra dimension so I'm just gonna grab the first one
# #     # a = np.expand_dims(a, 0)
# #     # # print('\n')
# #     print(a)
# #     # print('\n')
# #     #
# #     p0 = np.array([ [[237.,150.]],[[301.,104.]], [[272.,104.]], [[308.,210.]], [[264.,151.]], [[182.,186.]], [[175.,187.]]], np.float32)
# #     #
# #     # print('\n')
# #     # print(p0)
# #
# #
# #
# #
rachel1()
#
# #
# # assert len(x_head) == len(y_head) and len(x_head) == len(state)
# #
# # for x in range(len(x_head)):
# #     if state[x] != 'scenting' and state[x] != 'crawling':
# #         del x_head[x]
# #         del y_head[x]
# #         del state[x]
# #         print(x)
# #
# # print(len(x_head))
# #
# #
# #
# #
# #
# #
# ## params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 300,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# # #
# #
