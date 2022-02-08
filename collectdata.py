import cv2
from cv2 import data, detail_AffineBestOf2NearestMatcher
import mediapipe as mp
import argparse

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from math import sqrt
import math

from random import randrange

import moviepy.editor as mpy  # DON'T KEEP MP , COZ MEDIAPIPE 
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the input video")
ap.add_argument("-o", "--output", required=False, help="path to the output video")
ap.add_argument("-s", "--fps", type=int, default=30, help= "set fps of output video")
ap.add_argument("-l", "--buffer", type=int, default=30, help="max buffer size")
ap.add_argument("-b", "--black", type=str, default=False, help="set black background")
ap.add_argument("-w", "--white", type=str, default=False, help="set white background")

# ap.add_argument("-a", "--audio", required=False, help="extracted audio")
ap.add_argument("-r", "--result", required=False, help="final result")

args = vars(ap.parse_args())
ptsTail1 = deque(maxlen=args["buffer"])

cap = cv2.VideoCapture(args["input"])
out = cv2.VideoWriter("edited.mov", cv2.VideoWriter_fourcc(*"MJPG"), \
    args["fps"], (int(cap.get(3)), int(cap.get(4))))

mp_pose = mp.solutions.pose
jids = [0,11,12,13,14,15,16,17,18,23,24,25,26,27,28,31,32]

rawdataDict = {}
for i in jids:
    rawdataDict[i] = {}
# d = {0: {}, 11: {}, 12: {}, 13: {}, 14: {}, 15: {}, 16: {}, 17: {}, 18: {}, 23: {}, 24: {}, 25: {}, 26: {}, 27: {}, 28: {}, 31: {}, 32: {}}

signalXYDict = {}
 

# draw hybrid skeleton (triangle + lines)
def drawHybridSkeleton(image, frameDataDict):

    def drawTriangleFilled(pt1, pt2, pt3, color):
        triangle_pnts = np.array([pt1, pt2, pt3])
        # Create a contour with filled color in it
        cv2.drawContours(image, [triangle_pnts],0,color,-1)
    
    head = (frameDataDict[0][0],frameDataDict[0][1]-40 )
    color = (randrange(255), randrange(255), randrange(255))
    drawTriangleFilled(frameDataDict[8], frameDataDict[4], head, color)
    drawTriangleFilled(frameDataDict[7], frameDataDict[3], head, color)
    drawTriangleFilled(frameDataDict[10], frameDataDict[11], frameDataDict[12], color)
    drawTriangleFilled(frameDataDict[11], frameDataDict[15], frameDataDict[16],color )
    drawTriangleFilled(frameDataDict[8], frameDataDict[12], frameDataDict[16], color)
    drawTriangleFilled(frameDataDict[7], frameDataDict[11], frameDataDict[9], color)
    cv2.line(image, frameDataDict[3], frameDataDict[9], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[4], frameDataDict[10], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[9], frameDataDict[10], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[9], frameDataDict[11], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[12], frameDataDict[16], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[7], frameDataDict[15], (255, 255, 255), 1, lineType=cv2.LINE_AA)

# draw opacity
def opacityart(image, frameDataDict):
    overlay = image.copy()

    # manipulate overlay
    drawHybridSkeleton(overlay, frameDataDict)
    
    alpha = 0.6
    beta = 1-alpha
    cv2.addWeighted(overlay, alpha, image, beta, 0.0 , image)

# opacity with trail
def opacityWithTrail(image, frameDataDict, RFN):
    overlay = image.copy()

    TrailList = deque(maxlen=args["buffer"])
    TrailList2 = deque(maxlen=args["buffer"])
    TrailList3 = deque(maxlen=args["buffer"])

    for i in range(0, RFN):
            TrailList.appendleft(signalXYDict[0][i])
            TrailList2.appendleft(signalXYDict[17][i])
            TrailList3.appendleft(signalXYDict[18][i])
    
    for i in range(0, len(TrailList)):
            if TrailList[i] is not None and TrailList[i-1] is not None:  # This is for the loop animation
            #if i > 0:  # This is for the Trail curve only.

                # r = abs(TrailList[i-1][0] - TrailList[i-1][0])  # only a dot instead of circle
                r = int(abs(TrailList[i-1][0] - TrailList[i][0])/8)
                #cv2.line(image2, TrailList[i], TrailList[i-1], (255, 255, 255), 2, lineType=cv2.LINE_AA)
                #cv2.line(image2, TrailList[i], TrailList[i-1], (0, 0, 0), 1, lineType=cv2.LINE_AA)
                
                cv2.line(image2, TrailList2[i], TrailList2[i-1], (255, 255, 255), 10, lineType=cv2.LINE_AA)
                # cv2.line(image2, TrailList2[i], TrailList2[i-1], (0, 0, 0), 1, lineType=cv2.LINE_AA)

                cv2.line(image2, TrailList3[i], TrailList3[i-1], (255, 255, 255), 10, lineType=cv2.LINE_AA)
                
                # cv2.line(image2, TrailList3[i], TrailList3[i-1], (0, 0, 0), 1, lineType=cv2.LINE_AA)

                #cv2.circle(image2, (TrailList[i-1][0]+5,TrailList[i][1]), r, (255, 255, 255), -1)
                # cv2.line(image2, (TrailList[i][0]+5,TrailList[i][1]), (TrailList[i-1][0]+5,TrailList[i][1]) , (255, 255, 255), 1, lineType=cv2.LINE_AA)  # those dashed lines
                #cv2.line(image2, TrailList2[i], TrailList2[i-1], (255, 255, 255), 1, lineType=cv2.LINE_AA)
                #cv2.circle(image2, (TrailList2[i-1][0]+5,TrailList2[i][1]), r, (255, 255, 255), 1)
                #cv2.circle(image2, (TrailList3[i-1][0]+5,TrailList3[i][1]), r, (255, 255, 255), 1)
                # cv2.line(image2, (TrailList2[i][0]+5,TrailList2[i][1]), (TrailList2[i-1][0]+5,TrailList2[i][1]) , (255, 255, 255), 1, lineType=cv2.LINE_AA)
    TrailList.clear()
    TrailList2.clear()
    TrailList3.clear()

    alpha = 0.5
    beta = 1-alpha
    cv2.addWeighted(overlay, alpha, image, beta, 0.0 , image)

# function for skeleton
def opacityWithSkeleton(image, frameDataDict):
    """
    input: frameDataDict -> {0: (292, 289), 1: (326, 291), 2: (296, 296), 3: (320, 285), .... 16:(x,y)}
    """
    overlay = image.copy()

    head = (frameDataDict[0][0],frameDataDict[0][1]-40 )
    cv2.line(image, head, frameDataDict[3], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, head, frameDataDict[4], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, head, frameDataDict[7], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, head, frameDataDict[8], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[4], frameDataDict[8], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[3], frameDataDict[7], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[3], frameDataDict[9], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[4], frameDataDict[10], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[9], frameDataDict[10], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[10], frameDataDict[12], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[9], frameDataDict[11], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    #  cv2.line(image, frameDataDict[10], frameDataDict[11], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    # cv2.line(image, frameDataDict[12], frameDataDict[11], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[12], frameDataDict[16], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[11], frameDataDict[15], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[15], frameDataDict[16], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[15], frameDataDict[12], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    # cv2.line(image, frameDataDict[10], frameDataDict[11], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[16], frameDataDict[9], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[8], frameDataDict[12], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[8], frameDataDict[16], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[7], frameDataDict[11], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(image, frameDataDict[7], frameDataDict[15], (255, 255, 255), 1, lineType=cv2.LINE_AA)
    
    alpha = 0.5
    beta = 1-alpha
    cv2.addWeighted(overlay, alpha, image, beta, 0.0 , image)


# cryptoPunk


def Body(image, frameDataDict):

    rarm=(frameDataDict[2],frameDataDict[4])
    larm=(frameDataDict[1],frameDataDict[3])
    rhand=(frameDataDict[4],frameDataDict[6])
    lhand=(frameDataDict[3],frameDataDict[5])
    rpalm=(frameDataDict[6],frameDataDict[8])
    lpalm=(frameDataDict[5],frameDataDict[7])
    rthigh=(frameDataDict[10],frameDataDict[12])
    lthigh=(frameDataDict[9],frameDataDict[11])
    rnali=(frameDataDict[12],frameDataDict[14])
    lnali=(frameDataDict[11],frameDataDict[13])
    rfoot=(frameDataDict[14],frameDataDict[16])
    lfoot=(frameDataDict[13],frameDataDict[15])

    # listForEllipses=[rarm,larm,rhand,lhand,rpalm,lpalm,rthigh,lthigh,rnali,lnali,rfoot,lfoot]
    
    # for tutorial boogaloo
    listForEllipses=[rarm,larm,rhand,lhand,rpalm,lpalm,rthigh,lthigh,rnali,lnali,rfoot,lfoot]
    listForChestBelly=[frameDataDict[1],frameDataDict[2],frameDataDict[9],frameDataDict[10]]
    
    def centerPoint(j1, j2):
        """
        j1 = (x, y), j2 = (x, y)
        """
        return (int((j1[0]+j2[0])/2), int((j1[1]+j2[1])/2))
    def lenAxes(j1, j2):
        lenMajorAxis = math.sqrt((j1[0]-j2[0])**2 + (j1[1]-j2[1])**2)
        lenMajorAxis = int(lenMajorAxis/2)
        lenMinorAxis = int(lenMajorAxis/3)
        return (lenMajorAxis, lenMinorAxis)
    def Angle(j1, j2):
        angle = int(math.atan2((j1[1]-j2[1]),(j1[0]-j2[0]))*180/math.pi)
        return angle
    def drawEllipse(takeaTuple):
        """
        Take a tuple of points ((x1,y1),(x2,y2))
        """
        cc = centerPoint(takeaTuple[0],takeaTuple[1])
        axesLengths = lenAxes(takeaTuple[0],takeaTuple[1])
        theta = Angle(takeaTuple[0],takeaTuple[1])
        cv2.ellipse(image, cc, axesLengths, theta,
                              startAngle=0, endAngle=360, color=(200,200,200), thickness=1)
    
    def drawLine(pt1, pt2, color):
        cv2.line(image, pt1 , pt2, color, 1)

    def drawChestBelly(listOf4Points):
       """
       listOf4Points : [(x,y),(x,y),(x,y),(x,y)]
       """
       p0 = listOf4Points[0]
       p1 = listOf4Points[1]
       p2 = listOf4Points[2]
       p3 = listOf4Points[3]
   
       drawLine(p0, p1, (255, 255, 255))
       drawLine(p1, p2, (255, 255, 255))
       drawLine(p2, p3, (255, 255, 255))
       drawLine(p3, p0, (255, 255, 255))
    
    for i in listForEllipses:
        drawEllipse(i)
    drawChestBelly(listForChestBelly)
    
    # draw the punk head on the image
    headPoint = (frameDataDict[0])
    topY = headPoint[1]-70
    buttomY = headPoint[1]+70
    leftX = headPoint[0]-70
    rightX = headPoint[0]+70

    punk = cv2.imread("ape.png")
    resized_punk = cv2.resize(punk, (140, 140), interpolation=cv2.INTER_AREA)
    roi = image[topY:buttomY,leftX:rightX]
    patched = cv2.addWeighted(roi, 1, resized_punk, 0.9, 0)
    image[topY:buttomY,leftX:rightX] = patched
    # https://unpkg.com/browse/cryptopunk-icons@1.1.0/app/assets/

    overlay = image.copy()  # this is my raw image now with the ape.

    # manipulate overlay
    drawHybridSkeleton(image, frameDataDict)  # manipulated image

    # draw trails
    '''
    TrailList = deque(maxlen=args["buffer"])
    TrailList2 = deque(maxlen=args["buffer"])
    TrailList3 = deque(maxlen=args["buffer"])

    for i in range(0, RFN):
            TrailList.appendleft(signalXYDict[0][i])
            TrailList2.appendleft(signalXYDict[17][i])
            TrailList3.appendleft(signalXYDict[18][i])
    
    for i in range(0, len(TrailList)):
            # if TrailList[i] is not None and TrailList[i-1] is not None:  # This is for the loop animation
            if i > 0:  # This is for the Trail curve only.

                # r = abs(TrailList[i-1][0] - TrailList[i-1][0])  # only a dot instead of circle
                r = int(abs(TrailList[i-1][0] - TrailList[i][0])/8)
                #cv2.line(image2, TrailList[i], TrailList[i-1], (255, 255, 255), 2, lineType=cv2.LINE_AA)
                #cv2.line(image2, TrailList[i], TrailList[i-1], (0, 0, 0), 1, lineType=cv2.LINE_AA)
                
                cv2.line(image2, TrailList2[i], TrailList2[i-1], (255, 255, 255), 10, lineType=cv2.LINE_AA)
                # cv2.line(image2, TrailList2[i], TrailList2[i-1], (0, 0, 0), 1, lineType=cv2.LINE_AA)

                cv2.line(image2, TrailList3[i], TrailList3[i-1], (255, 255, 255), 10, lineType=cv2.LINE_AA)
                
                
    TrailList.clear()
    TrailList2.clear()
    TrailList3.clear()
    '''
    alpha = 0.8
    beta = 1-alpha
    cv2.addWeighted(overlay, alpha, image, beta, 0.0 , image)



if __name__== "__main__":

    fps = cap.get(cv2.CAP_PROP_FPS)
    totalNumFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(totalNumFrames)
    # print("fps : {0}".format(fps))
    totalframes = 0

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.6
    ) as pose:
    
        while cap.isOpened():
            success, image = cap.read()
            frameNumber = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # print(frameNumber)

            if not success:
                print("-"*20)
                print("Ignoring empty camera frame")
                print("-"*20)
                # for live video use continue instead of break.
                break
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            # print(results)
            # print(results.pose_landmarks.landmark)
            if results.pose_landmarks:
                for jointid, lm in enumerate(results.pose_landmarks.landmark):
                    for i in jids:
                        if jointid == i:
                            h, w, c = image.shape
                            cx, cy = int(lm.x*w), int(lm.y*h)
                            rawdataDict[i][frameNumber] = (cx,cy)
            else:
                print("no landmark for this frame")
                # store the values of the joints as of the previous frame in the dictionary
                for i in jids:
                    rawdataDict[i][frameNumber] = rawdataDict[i][frameNumber-1]  # assuming the first frame always should have a landmark, fixme.
                #print(rawdataDict)
            
            cv2.imshow('GrooveCanvas', image)
            totalframes += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break

    # cap.release()

    # cv related with loop works ends here.
    # ------------------ now we r inside main function ---------
    # print(rawdataDict)  # {0:{1:(245,341), 2:(246,342), 3:(249,347)}, 11:{1:(245,341),2:(246,342),3:(249,347)}}
    # print(rawdataDictWithoutFrameNumber)  # {0:[(cxf1,cyf1),(cxf113,cyf113)], 11:[(cxf1,cyf1),(cxf113,cyf113)], ...}
    
    print("-"*20)
    print("totalframes: ", totalframes)
    k0 = len(list(rawdataDict[0].keys()))
    k11 = len(list(rawdataDict[11].keys()))
    k31 = len(list(rawdataDict[31].keys()))
    # print("k0 is : ", k0)
    # print("k11 is : ", k11)
    # print("k31 is : ", k31)
    
    for jid in jids:
        signalXY = []
        signalJoint = rawdataDict[jid]
        for key, value in signalJoint.items():
            signalXY.append(value)
        
        signalX = []
        signalY = []

        for i in signalXY:
            signalX.append(i[0])
            signalY.append(i[1])
        
        b, a = signal.butter(4, 2/12, 'low')  # 2/(fps/2), 2/fps/5 -> 0.2 sec window
        butterX = signal.filtfilt(b, a, signalX)
        butterY = signal.filtfilt(b, a, signalY)
        butterX = butterX.tolist()
        butterY = butterY.tolist()
        
        for i in range(0, len(butterX)):  
            butterX[i] = round(butterX[i])
            butterY[i] = round(butterY[i])  # len(xbutter) = len(ybutter)

        # concatenate
        coordinates = []
        for i in range (0,len(butterX)):           
            coordinates.append((butterX[i], butterY[i]))
        signalXYDict[jid] = coordinates

        '''
        plt.plot(signalX, 'k-', label='input')
        plt.plot(signalY, 'k-', label='input')
        plt.plot(butterX, 'b-', linewidth=1, label='gust')
        plt.plot(butterY, 'b-', linewidth=1, label='gust')
        plt.legend(loc='best')
        plt.show()
        '''

        

        
        
    # print(signalXYDict)  # This is after signal processing.
    
    # convert the data to per frame basis config:
    # {
    #  1:{0:(cx,cy), 11:(cx,cy), 12(cx,cy),...},
    #  2:{0:(cx,cy), 11:(cx,cy), 12(cx,cy),...}
    # }
    
    # ----
    # for Trail data
    # print(signalXYDict[0])
    '''
    TrailList = deque(maxlen=args["buffer"])
    TrailList2 = deque(maxlen=args["buffer"])
    TrailList3 = deque(maxlen=args["buffer"])
    '''
    # ----

    keyList = list(signalXYDict.keys())
    valueList = list(signalXYDict.values())
    # print(len(valueList))
    # print(valueList[0])
    # print(valueList[12])

    dataForRendering = {}
    for i in range(1,len(valueList[0])+1):
        perFrameData = {}
        for j in range(0,len(keyList)):
            perFrameData[j] = valueList[j][i-1]
        # print(perFrameData)
        # print("------")
        dataForRendering[i] = perFrameData
    # print(dataForRendering)


    # ---------------open read the video again for rendering this time ---------------
    
    cap2 = cv2.VideoCapture(args["input"])
    while cap2.isOpened():
        success, image2 = cap2.read()
        # RFN : render frame number
        RFN = int(cap2.get(cv2.CAP_PROP_POS_FRAMES))
        # print(RFN)
        if not success:
          print("Ignoring empty camera frame.")
          break

        image2 = cv2.cvtColor(cv2.flip(image2, 1), cv2.COLOR_BGR2RGB)
        image2.flags.writeable = False
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        # image2 = cv2.xphoto.oilPainting(image2, 5, 1)
        # image2 = cv2.stylization(image2, sigma_s=190, sigma_r=0.4)

        if args["black"]:
            image2 = image2*0
        
        if args["white"]:
            image2 = image2*255
        # print(dataForRendering[RFN])
        # drawHybridSkeleton(image2, dataForRendering[RFN])
        # opacityart(image2, dataForRendering[RFN] )
        # opacityWithTrail(image2, dataForRendering[RFN], RFN)
        opacityWithSkeleton(image2, dataForRendering[RFN])
        # Body(image2, dataForRendering[RFN])
        
        '''
        for i in range(0, RFN):
            TrailList.appendleft(signalXYDict[0][i])
            TrailList2.appendleft(signalXYDict[17][i])
            TrailList3.appendleft(signalXYDict[18][i])
        # print(TrailList)

        # draw Trail here
        for i in range(0, len(TrailList)):
            #if TrailList[i] is not None and TrailList[i-1] is not None:  # This is for the loop animation
            if i > 0:  # This is for the Trail curve only.

                # r = abs(TrailList[i-1][0] - TrailList[i-1][0])  # only a dot instead of circle
                r = int(abs(TrailList[i-1][0] - TrailList[i][0])/2)
                #cv2.line(image2, TrailList[i], TrailList[i-1], (255, 255, 255), 2, lineType=cv2.LINE_AA)
                #cv2.line(image2, TrailList[i], TrailList[i-1], (0, 0, 0), 1, lineType=cv2.LINE_AA)
                
                cv2.line(image2, TrailList2[i], TrailList2[i-1], (255, 255, 255), 2, lineType=cv2.LINE_AA)
                # cv2.line(image2, TrailList2[i], TrailList2[i-1], (0, 0, 0), 1, lineType=cv2.LINE_AA)

                cv2.line(image2, TrailList3[i], TrailList3[i-1], (255, 255, 255), 2, lineType=cv2.LINE_AA)
                
                # cv2.line(image2, TrailList3[i], TrailList3[i-1], (0, 0, 0), 1, lineType=cv2.LINE_AA)

                #cv2.circle(image2, (TrailList[i-1][0]+5,TrailList[i][1]), r, (255, 255, 255), -1)
                # cv2.line(image2, (TrailList[i][0]+5,TrailList[i][1]), (TrailList[i-1][0]+5,TrailList[i][1]) , (255, 255, 255), 1, lineType=cv2.LINE_AA)  # those dashed lines
                #cv2.line(image2, TrailList2[i], TrailList2[i-1], (255, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.circle(image2, (TrailList2[i-1][0]+5,TrailList2[i][1]), r, (255, 255, 255), 1)
                cv2.circle(image2, (TrailList3[i-1][0]+5,TrailList3[i][1]), r, (255, 255, 255), 1)
                # cv2.line(image2, (TrailList2[i][0]+5,TrailList2[i][1]), (TrailList2[i-1][0]+5,TrailList2[i][1]) , (255, 255, 255), 1, lineType=cv2.LINE_AA)
        TrailList.clear()
        TrailList2.clear()
        TrailList3.clear()
        '''

        cv2.imshow('GrooveCanvas', image2)
        out.write(image2)
                        
        # go and read the next frame
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap2.release()
    out.release()

    # extract the music here.
    video = mpy.VideoFileClip(args["input"])
    audio = video.audio
    audio.write_audiofile("audio.mp3")  # args["audio"]

    # mix and output the final video
    edited = mpy.VideoFileClip("edited.mov")
    edited.write_videofile(args["result"], codec='libx264', audio_codec='aac', audio="audio.mp3", remove_temp=True)
