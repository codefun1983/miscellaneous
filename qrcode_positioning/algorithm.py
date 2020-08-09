import cv2
import numpy as np

frame = cv2.imread(r'<your image path>')
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,80,120)
#find qrcode three points
contours , hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]
found = []
for i in range(len(contours)):
    k = i
    c = 0
    while hierarchy[k][2] != -1:
        k = hierarchy[k][2]
        c = c+1
    if c >= 5:
        found.append(i)
# find center of gravity
cores = []
for i in found:
    mm = cv2.moments(contours[i+5])
    # if mm['m00'] == 0:
    #     return error
    core = (int(mm['m10']/mm['m00']) , int(mm['m01']/mm['m00']))
    cores.append(list(core))
# three points of polarization check
if len(cores) > 2:
    correct = []
    correct2 = []
    correct3 = []
    correct4 = []
    img = cv2.GaussianBlur(frame, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    opens = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    hits = cv2.morphologyEx(opens, cv2.MORPH_HITMISS, kernel, iterations=2)
    thresh = cv2.threshold(hits, 120, 255, cv2.THRESH_BINARY)
    hits2 = cv2.morphologyEx(opens, cv2.MORPH_HITMISS, kernel, iterations=45)
    thresh2 = cv2.threshold(hits2, 120, 255, cv2.THRESH_BINARY)
# 2 types to figure out valid point from image algorithm
    for x in range(len(cores)):
        for i in range(len(cores) - x - 1):
            k = []
            k2 = []
            kkk = []
            kkk2 = []
            kc = 0
            kc1 = 0
            k2c = 0
            k2c1 = 0
            r = np.sqrt(
                (cores[i][0] - cores[i + (x + 1)][0]) ** 2 + (cores[i][1] - cores[i + (x + 1)][1]) ** 2)
            for lp in np.linspace(cores[i], cores[i + (x + 1)], int(r)):
                lp = [int(lp[0]), int(lp[1])]
                k.append(thresh[1][int(lp[0])][int(lp[1])])
                k2.append(thresh2[1][int(lp[0])][int(lp[1])])
            for j in k:
                if j == 0:
                    kc = kc + 1
                    kkk.append(kc)
                    kc1 = 0
                if j == 255:
                    kc1 = kc1 + 1
                    kkk.append(kc1)
                    kc = 0
            for jj in k2:
                if jj == 0:
                    k2c = k2c + 1
                    kkk2.append(k2c)
                    k2c1 = 0
                if jj == 255:
                    k2c1 = k2c1 + 1
                    kkk2.append(k2c1)
                    k2c = 0
            for rd in range(len(kkk)):
                if kkk[rd] - kkk[rd - 1] == 1:
                    kkk[rd - 1] = 0
            for rd2 in range(len(kkk2)):
                if kkk2[rd2] - kkk2[rd2 - 1] == 1:
                    kkk2[rd2 - 1] = 0
            delete = [0, 1, 2, 3, 4]
            kkk = [i for i in kkk if i not in delete]
            kkk2 = [i for i in kkk2 if i not in delete]
            if np.var(kkk) < 2:
                correct.append(cores[i])
                correct.append(cores[i + (x + 1)])
            elif np.var(kkk2) < 2:
                correct2.append(cores[i])
                correct2.append(cores[i + (x + i)])
            else:
                # return error
                pass
# 1 type figure out valid point from math algorithm
    # combinations
    cb = []
    gyi = len(cores)
    for i in range(gyi):
        carray = np.ones(3, int)
        cb.append(carray * (i + 1))
    darray = np.append(cb[0], cb[1])
    for i in range(gyi - 1 - 1):
        darray = np.append(darray, cb[i + 2])
    garray = darray.reshape(3, gyi)
    garray = np.transpose(garray)
    gapp = []
# caculate
    from math import factorial as fact
    ff = fact(gyi) / (fact(3) * fact(gyi - 3))
    if ff < 11:  # over 5 point return error
        for i in range(gyi):
            for ii in range(3):
                for iii in range(gyi):
                    if garray[i][ii] == darray[(3 * iii) + 1]:
                        gapp.append(cores[iii])  # transform matrix to coordinate
    else:
        # return error
        pass
    ppp = []
    for i in range(len(garray)):
        kfc = gapp[(3 * i):(3 * i + 3)]
        ppp.append(kfc)
    correct_rate = []
    crmin = lambda cr0, cr1, cr2: cr0 if (cr0 < cr1 and cr0 < cr2) else (cr1 if (cr1 < cr0 and cr1 < cr2) else cr2)
    for i in range(len(ppp)):
        core_rate = []
        core_angle = []
        core_distance = []
        pppp = ppp[i]
        for i in range(len(pppp) - 1):
            for ii in range(len(pppp) - 1 - i):
                core_distance.append(np.sqrt((pppp[i][0] - pppp[ii + 1 + i][0]) ** 2 + (pppp[i][1] - pppp[ii + 1 + i][1]) ** 2))
        for i in range(len(core_distance) - 1):
            for ii in range(len(core_distance) - 1 - i):
                ee = core_distance[i]
                tt = core_distance[1 + i + ii]
                n = [ee, tt][tt < ee] / [ee, tt][tt > ee]
                core_angle.append(n)
        for i in range(len(core_angle) - 1):
            for ii in range(len(core_angle) - 1 - i):
                ee = core_angle[i]
                tt = core_angle[1 + i + ii]
                n = ([ee, tt][tt > ee] - [ee, tt][tt < ee]) / (([ee, tt][tt > ee] + [ee, tt][tt < ee]) / 2)
                core_rate.append(n)
        cr0 = core_rate[0]
        cr1 = core_rate[1]
        cr2 = core_rate[2]
        n = [i for i in core_rate if i not in [crmin(cr0, cr1, cr2)]]  # ignore the minimum value
        nn = ([n[0], n[1]][n[1] > n[0]] - [n[0], n[1]][n[1] < n[0]]) / (([n[0], n[1]][n[1] > n[0]] + [n[0], n[1]][n[1] < n[0]]) / 2)
        if nn < 0.1:
            if pppp not in correct_rate:
                correct_rate.append(pppp)
        else:
            # return error
            pass
    for i in correct:
        if i not in correct3:
            correct3.append(i)
    for i in correct2:
        if i not in correct4:
            correct4.append(i)
    # if correct_rate == []:
    #     pass
    correct6 = [i for i in correct_rate[0]]
    # correct6 = [np.ndarray.tolist(i) for i in core_rate[0]]
    len1 = len(correct3)
    len2 = len(correct4)
    len3 = len(correct6)
    if len1 == 3 and len2 == 3 and len3 == 3:
        try:
            if sorted(correct3) == sorted(correct4):
                final = correct4
            elif sorted(correct6) == sorted(correct3) or sorted(correct6) == sorted(correct4):
                final = correct6
        except:
            pass
    elif len1 == 3 and len2 == 3:
        try:
            if sorted(correct3) == sorted(correct4):
                final = correct4
        except:
            pass
    elif len1 == 3 and len3 == 3:
        try:
            if sorted(correct3) == sorted(correct6):
                final = correct6
        except:
            pass
    elif len2 == 3 and len3 == 3:
        try:
            if sorted(correct4) == sorted(correct6):
                final = correct6
        except:
            pass
    else:
        final = 0
    if final != 0:
        x0 = final[0][0]
        y0 = final[0][1]
        x1 = final[1][0]
        y1 = final[1][1]
        x2 = final[2][0]
        y2 = final[2][1]
        line_bc = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        line_ab = np.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2)
        line_ac = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        a = [line_ac, line_ab][line_ab < line_ac] / [line_ac, line_ab][line_ab > line_ac]
        b = [line_bc, line_ab][line_ab < line_bc] / [line_bc, line_ab][line_ab > line_bc]
        c = [line_bc, line_ac][line_ac < line_bc] / [line_bc, line_ac][line_ac > line_bc]
        ab = ([a, b][b > a] - [a, b][b < a]) / (([a, b][b > a] + [a, b][b < a]) / 2)
        ac = ([a, c][c > a] - [a, c][c < a]) / (([a, c][c > a] + [a, c][c < a]) / 2)
        bc = ([b, c][c > b] - [b, c][c < b]) / (([b, c][c > b] + [b, c][c < b]) / 2)
        regulate_point = []
        if crmin(ab,ac,bc) == ab:
            middle = [int(x0+x2)/2 , int(y0+y2)/2]
            if x1 >= middle[0] and y1 >= middle[1]:
                quadrant = 4
                if x0 >= middle[0] and y0 <= middle[1]:
                    regulate_point = [final[0],final[1],final[2]]
                else:
                    regulate_point = [final[2], final[1], final[0]]
            elif x1 <= middle[0] and y1 <= middle[1]:
                quadrant = 2
                if x0 <= middle[0] and y0 >= middle[1]:
                    regulate_point = [final[0], final[1], final[2]]
                else:
                    regulate_point = [final[2], final[1], final[0]]
            elif x1 <= middle[0] and y1 >= middle[1]:
                quadrant = 3
                if x0 >= middle[0] and y0 >= middle[1]:
                    regulate_point = [final[0], final[1], final[2]]
                else:
                    regulate_point = [final[2], final[1], final[0]]
            else:
                quadrant = 1
                if x0 <= middle[0] and y0 <= middle[1]:
                    regulate_point = [final[0], final[1], final[2]]
                else:
                    regulate_point = [final[2], final[1], final[0]]

        elif crmin(ab, ac, bc) == ac:
            middle = [int(x1 + x2) / 2, int(y1 + y2) / 2]
            if x0 >= middle[0] and y0 >= middle[1]:
                quadrant = 4
                if x1 >= middle[0] and y1 <= middle[1]:
                    regulate_point = [final[1], final[0], final[2]]
                else:
                    regulate_point = [final[2], final[0], final[1]]
            elif x0 <= middle[0] and y0 <= middle[1]:
                quadrant = 2
                if x1 <= middle[0] and y1 >= middle[1]:
                    regulate_point = [final[1], final[0], final[2]]
                else:
                    regulate_point = [final[2], final[0], final[1]]
            elif x0 <= middle[0] and y0 >= middle[1]:
                quadrant = 3
                if x1 >= middle[0] and y1 >= middle[1]:
                    regulate_point = [final[1], final[0], final[2]]
                else:
                    regulate_point = [final[2], final[0], final[1]]
            else:
                quadrant = 1
                if x1 <= middle[0] and y1 <= middle[1]:
                    regulate_point = [final[1], final[0], final[2]]
                else:
                    regulate_point = [final[2], final[0], final[1]]

        else:
            middle = [int(x0 + x1) / 2, int(y0 + y1) / 2]
            if x2 >= middle[0] and y2 >= middle[1]:
                quadrant = 4
                if x0 >= middle[0] and y0 <= middle[1]:
                    regulate_point = [final[0], final[2], final[1]]
                else:
                    regulate_point = [final[1], final[2], final[0]]
            elif x2 <= middle[0] and y2 <= middle[1]:
                quadrant = 2
                if x0 <= middle[0] and y0 >= middle[1]:
                    regulate_point = [final[0], final[2], final[1]]
                else:
                    regulate_point = [final[1], final[2], final[0]]
            elif x2 <= middle[0] and y2 >= middle[1]:
                quadrant = 3
                if x0 >= middle[0] and y0 >= middle[1]:
                    regulate_point = [final[0], final[2], final[1]]
                else:
                    regulate_point = [final[1], final[2], final[0]]
            else:
                quadrant = 3
                if x0 <= middle[0] and y0 <= middle[1]:
                    regulate_point = [final[0], final[2], final[1]]
                else:
                    regulate_point = [final[1], final[2], final[0]]
    else:
        regulate_point = 0
        quadrant = 0
if  regulate_point == 0 and quadrant == 0:
    print('not detect')
    exit(0)
# find point four
fx = []
fy = []
point4 = []
for i in range(len(regulate_point)):
    fx.append(regulate_point[i][0])
    fy.append(regulate_point[i][1])
for i in range(2):
    point4.append((regulate_point[2][i])-(regulate_point[1][i])+(regulate_point[0][i]))
fx.append(point4[0])
fy.append(point4[1])
xmax = max(fx)
ymax = max(fy)
xmin = min(fx)
ymin = min(fy)
rate = 0.2
rows, cols = frame.shape[:2]
if quadrant % 2 != 0:
    d1 = np.sqrt((regulate_point[1][0] - regulate_point[0][0])**2 + (regulate_point[1][1] - regulate_point[0][1])**2)   # row
    d2 = np.sqrt((regulate_point[2][0] - regulate_point[1][0])**2 + (regulate_point[2][1] - regulate_point[1][1])**2)   # column
else:
    d1 = np.sqrt((regulate_point[2][0] - regulate_point[1][0])**2 + (regulate_point[2][1] - regulate_point[1][1])**2)   # row
    d2 = np.sqrt((regulate_point[1][0] - regulate_point[0][0])**2 + (regulate_point[1][1] - regulate_point[0][1])**2)   # column
if quadrant == 1:
    P4 = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax],[xmin-(d1*rate),ymin-(d2*rate)],[xmax+(d1*rate),ymin-(d2*rate)],[xmax+(d1*rate),ymax+(d2*rate)],[xmin-(d1*rate),ymax+(d2*rate)]]
    shape = [[0,0],[rows,0],[rows,cols],[0,cols]]
elif quadrant == 2:
    P4 = [[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax],[xmin-(d1*rate), ymax+(d2*rate)], [xmin-(d1*rate), ymin-(d2*rate)], [xmax+(d1*rate), ymin-(d2*rate)], [xmax+(d1*rate), ymax+(d2*rate)]]
    shape = [[0,cols],[0,0],[rows,0],[rows,cols]]
elif quadrant == 3:
    P4 = [[xmax, ymax], [xmin, ymax], [xmin, ymin], [xmax, ymin],[xmax+(d1*rate), ymax+(d2*rate)], [xmin-(d1*rate), ymax+(d2*rate)], [xmin-(d1*rate), ymin-(d2*rate)], [xmax+(d1*rate), ymin-(d2*rate)]]
    shape = [[rows,cols],[0,cols],[0,0],[rows,0]]
else:
    P4 = [[xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin],[xmax+(d1*rate), ymin-(d2*rate)], [xmax+(d1*rate), ymax+(d2*rate)], [xmin-(d1*rate), ymax+(d2*rate)], [xmin-(d1*rate), ymin-(d2*rate)]]
    shape = [[rows,0],[rows,cols],[0,cols],[0,0]]
##-------------------------------------affine transform
M = cv2.getAffineTransform(np.array(regulate_point[:3],np.float32),np.array(P4[:3],np.float32))
result = cv2.warpAffine(frame,M,(cols,rows))
##-------------------------------------perspective transform
MM = cv2.getPerspectiveTransform(np.float32(P4[4:]),np.float32(shape))
result2 = cv2.warpPerspective(result,MM,(rows,cols))