import cv2
import mediapipe as mp
import ssl
import argparse
ssl._create_default_https_context = ssl._create_unverified_context
import math
from math import sqrt, atan2
from collections import deque 
import numpy as np
from skimage.draw import line_aa
from skimage.util import img_as_ubyte, img_as_float

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# gebble it -------------------------------------------------------------------
# For webcam input:
# construct the argument parser and pass the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the input video")
ap.add_argument("-o", "--output", required=False, help="path to the output")
ap.add_argument("-s", "--fps", type=int, default=30, help= "set fps of output video")
ap.add_argument("-b", "--black", type=str, default=False, help="set black background")
ap.add_argument("-l", "--buffer", type=int, default=32,
	help="max buffer size")  # the maximum size of deque
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["input"])
out = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*"MJPG"), \
    args["fps"], (int(cap.get(3)), int(cap.get(4))))


pts = []  # global space holder for Trace(): art3
getCoordinatesListValue = [] # joint points coordinate.
jointids = [0,11,12,13,14,17,18,23,24,25,26,31,32]
#                               7  8  9 10 11  12

Art4jids = [0,11,12,13,14,15,16,17,18,23,24,25,26,27,28,31,32]
#                                     9, 10,11,12,13,14,15,16
getCoordinatesListValueArt4 = []
ptsTail1 = deque(maxlen=args["buffer"])
ptsTail2 = deque(maxlen=args["buffer"])

def drawLine(pt1, pt2, color):
    cv2.line(image, pt1 , pt2, color, 1)

def drawTriangleFilled(pt1, pt2, pt3, color):
    
    triangle_pnts = np.array([pt1, pt2, pt3])
    # Create a contour with filled color in it
    cv2.drawContours(image, [triangle_pnts],0,color,-1)

def drawTriangleFilled2(pt1, pt2, pt3, color):
    
    triangle_pnts = np.array([pt1, pt2, pt3])
    # Create a contour with filled color in it
    cv2.drawContours(image, [triangle_pnts],0,color,-1)

def drawTriangleFilled3(pt1, pt2, pt3, color):
    triangle_pnts = np.array([pt1, pt2, pt3])
    # Create a contour with filled color in it
    cv2.drawContours(image, [triangle_pnts],0,color,-1)

def drawTriangleFilled4(pt1, pt2, pt3, color):
    triangle_pnts = np.array([pt1, pt2, pt3])
    # Create a contour with filled color in it
    cv2.drawContours(image, [triangle_pnts],0,color,-1)

def drawTriangleContour(pt1, pt2, pt3):
    triangle_pnts = np.array([pt1, pt2, pt3])
    cv2.polylines(image, np.array([triangle_pnts]), True, (220,220,220), 1)

# A better approach would be to store the coordinates in a dictionary.
# art 2
def getCoordinatesList(jointidlist):
        """
        arg: jointidlist is a list of joint ids.
        """

        if results.pose_landmarks:

            # getCoordinatesListValue.clear()
            getCoordinatesListValueArt4.clear()
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                for x in jointidlist:
                    if id == x:
                        getCoordinatesListValueArt4.append((cx, cy))
            return getCoordinatesListValueArt4
        
        else:
            # to stop flickering
            return getCoordinatesListValueArt4

# art1
def drawRotoscopeAllConnections(jointList):
    for i in jointList:
        jointList.remove(i)
    
        for j in jointList:
            drawLine(i,j)

# art3 : Trace the joint
def Trace(jointid):
    if results.pose_landmarks:
            
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id == jointid:
                pts.append((cx, cy))
                for i in range(0, len(pts)):
                    if i > 0:
                        cv2.line(image, pts[i-1], pts[i], (255, 255, 255), 1)
    else:
        # to stop flickering
        for j in range(0, len(pts)):
            if j > 0:
                cv2.line(image, pts[j-1], pts[j], (255, 255, 255), 1)

# art 5 n art 6 ( glitch )
def TraceTail1(jointid):
    if results.pose_landmarks:
            
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id == jointid:
                ptsTail1.appendleft((cx, cy))
                for i in range(1, len(ptsTail1)):
                    # if either of th etracked points are none, ignore them.
                    if ptsTail1[i-1] is None or ptsTail1[i] is None:
                        continue
                    # otherwise compute thickness of line n draw connecting lines
                    # thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
                    cv2.line(image, ptsTail1[i-1], ptsTail1[i], (230, 230, 230), 3, lineType=cv2.LINE_AA)

def TraceTail2(jointid):
    if results.pose_landmarks:
            
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id == jointid:
                ptsTail2.appendleft((cx, cy))
                for i in range(1, len(ptsTail2)):
                    # if either of th etracked points are none, ignore them.
                    if ptsTail2[i-1] is None or ptsTail2[i] is None:
                        continue
                    # otherwise compute thickness of line n draw connecting lines
                    # thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
                    cv2.line(image, ptsTail2[i-1], ptsTail2[i], (230, 230, 230), 1, lineType=cv2.LINE_AA)

# art parallel lines for skate boarders
ptsP1 = deque(maxlen=args["buffer"])  # ptsParallelLine 1 : actual middle line
ptsP2 = deque(maxlen=args["buffer"])
ptsP3 = deque(maxlen=args["buffer"])
ptsP4 = deque(maxlen=args["buffer"])
ptsP5 = deque(maxlen=args["buffer"])

ptsP12 = deque(maxlen=args["buffer"])  # ptsParallelLine 1 : actual middle line
ptsP22 = deque(maxlen=args["buffer"])
ptsP32 = deque(maxlen=args["buffer"])
ptsP42 = deque(maxlen=args["buffer"])
ptsP52 = deque(maxlen=args["buffer"])

def parallalLines1(jointid):
    if results.pose_landmarks:
            
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id == jointid:
                ptsP1.appendleft((cx, cy))
                ptsP2.appendleft((cx-20, cy))
                ptsP3.appendleft((cx+20, cy))
                ptsP4.appendleft((cx-40, cy))
                ptsP5.appendleft((cx+40, cy))

                for i in range(1, len(ptsP1)):
                    # if either of th etracked points are none, ignore them.
                    if ptsP1[i-1] is None or ptsP1[i] is None:
                        continue
                    # otherwise compute thickness of line n draw connecting lines
                    # thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
                    cv2.line(image, ptsP1[i-1], ptsP1[i], (230, 230, 230), 1, lineType=cv2.LINE_AA)
                    cv2.line(image, ptsP2[i-1], ptsP2[i], (230, 230, 230), 1, lineType=cv2.LINE_AA)
                    cv2.line(image, ptsP3[i-1], ptsP3[i], (230, 230, 230), 1, lineType=cv2.LINE_AA)
                    cv2.line(image, ptsP4[i-1], ptsP4[i], (230, 230, 230), 1, lineType=cv2.LINE_AA)
                    cv2.line(image, ptsP5[i-1], ptsP5[i], (230, 230, 230), 1, lineType=cv2.LINE_AA)

def parallalLines2(jointid):
    if results.pose_landmarks:
            
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id == jointid:
                ptsP12.appendleft((cx, cy))
                ptsP22.appendleft((cx+20, cy))
                ptsP32.appendleft((cx-20, cy))
                ptsP42.appendleft((cx+40, cy))
                ptsP52.appendleft((cx-40, cy))

                for i in range(1, len(ptsP12)):
                    # if either of th etracked points are none, ignore them.
                    if ptsP12[i-1] is None or ptsP12[i] is None:
                        continue
                    # otherwise compute thickness of line n draw connecting lines
                    # thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
                    cv2.line(image, ptsP12[i-1], ptsP12[i], (0, 0, 0), 1, lineType=cv2.LINE_AA)
                    cv2.line(image, ptsP22[i-1], ptsP22[i], (0, 0, 0), 1, lineType=cv2.LINE_AA)
                    cv2.line(image, ptsP32[i-1], ptsP32[i], (0, 0, 0), 1, lineType=cv2.LINE_AA)
                    #cv2.line(image, ptsP42[i-1], ptsP42[i], (0, 0, 0), 1, lineType=cv2.LINE_AA)
                    #cv2.line(image, ptsP52[i-1], ptsP52[i], (0, 0, 0), 1, lineType=cv2.LINE_AA)

# art 4 ellipse
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
def drawChestBelly(listOf4Points):
    """
    listOf4Points : [(x,y),(x,y),(x,y),(x,y)]
    """
    p0 = listOf4Points[0]
    p1 = listOf4Points[1]
    p2 = listOf4Points[2]
    p3 = listOf4Points[3]

    drawLine(p0, p1)
    drawLine(p1, p2)
    drawLine(p2, p3)
    drawLine(p3, p0)

# art 6
def drawConsciousness(jointid=[0,11,12]):
    list1 = []
    if results.pose_landmarks:
            
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id == jointid[0]:
                list1.append((cx, cy))
            elif id == jointid[1]:
                list1.append((cx,cy))
            elif id == jointid[2]:
                list1.append((cx,cy))
            
            if len(list1) == 3:
                cc = list1[0]
                lenMajorAxis = math.sqrt((list1[1][0]-list1[2][1])**2 + (list1[1][0]-list1[2][1])**2)
                lenMajorAxis = int(lenMajorAxis/4)
                lenMinorAxis = lenMajorAxis
                axesLengths = (lenMajorAxis, lenMinorAxis)
                axesLengths2 = (lenMajorAxis+10, lenMinorAxis+10)
                theta = Angle(list1[1],list1[2])
                cv2.ellipse(image, cc, axesLengths, theta,
                                      startAngle=0, endAngle=270, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)  # (230,230,230) (0,0,255)
                cv2.ellipse(image, cc, axesLengths2, theta,
                                      startAngle=90, endAngle=360, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)  # (30,30,30) (0,255,0)

def Body(getCoordinatesListValueArt4):

    # rarm=(getCoordinatesListValueArt4[2],getCoordinatesListValueArt4[4])
    # larm=(getCoordinatesListValueArt4[1],getCoordinatesListValueArt4[3])
    # rhand=(getCoordinatesListValueArt4[4],getCoordinatesListValueArt4[6])
    # lhand=(getCoordinatesListValueArt4[3],getCoordinatesListValueArt4[5])
    # rpalm=(getCoordinatesListValueArt4[6],getCoordinatesListValueArt4[8])
    # lpalm=(getCoordinatesListValueArt4[5],getCoordinatesListValueArt4[7])
    rthigh=(getCoordinatesListValueArt4[10],getCoordinatesListValueArt4[12])
    lthigh=(getCoordinatesListValueArt4[9],getCoordinatesListValueArt4[11])
    rnali=(getCoordinatesListValueArt4[12],getCoordinatesListValueArt4[14])
    lnali=(getCoordinatesListValueArt4[11],getCoordinatesListValueArt4[13])
    rfoot=(getCoordinatesListValueArt4[14],getCoordinatesListValueArt4[16])
    lfoot=(getCoordinatesListValueArt4[13],getCoordinatesListValueArt4[15])

    # listForEllipses=[rarm,larm,rhand,lhand,rpalm,lpalm,rthigh,lthigh,rnali,lnali,rfoot,lfoot]
    
    # for tutorial boogaloo
    listForEllipses=[rthigh,lthigh,rnali,lnali,rfoot,lfoot]
    listForChestBelly=[getCoordinatesListValueArt4[13],getCoordinatesListValueArt4[13],
                      getCoordinatesListValueArt4[13],getCoordinatesListValueArt4[13]]

    for i in listForEllipses:
        drawEllipse(i)

    drawChestBelly(listForChestBelly)

    '''
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    '''
# art2

    """
    jointList: pass the actual list of joint coordinates 
    """
    
    # drawLine(jointList[0], jointList[1])
    # drawLine(jointList[0], jointList[2])
    # drawLine(jointList[1], jointList[2])
    '''
    drawLine(jointList[0], jointList[3])
    drawLine(jointList[0], jointList[4])
    drawLine(jointList[0], jointList[5])
    drawLine(jointList[0], jointList[6])
    '''
    # drawLine(jointList[2], jointList[4])
    # drawLine(jointList[4], jointList[6])
    # drawLine(jointList[1], jointList[3])
    # drawLine(jointList[3], jointList[5])
    
    # drawLine(jointList[4], jointList[8])
    # drawLine(jointList[4], jointList[10])
    # drawLine(jointList[6], jointList[10])
    # drawLine(jointList[6], jointList[12])
    # drawLine(jointList[3], jointList[7])
    # drawLine(jointList[3], jointList[9])
    # drawLine(jointList[5], jointList[9])
    # drawLine(jointList[5], jointList[11])
    """
    drawLine(jointList[11], jointList[12])
    drawLine(jointList[9], jointList[10])
    drawLine(jointList[7], jointList[8])

    drawLine(jointList[8], jointList[9])
    drawLine(jointList[9], jointList[12])

    drawLine(jointList[8], jointList[10])
    drawLine(jointList[10], jointList[12])
    drawLine(jointList[7], jointList[9])
    drawLine(jointList[9], jointList[11])
    """
    # headspin1
    """
    drawLine(jointList[11], jointList[12])
    drawLine(jointList[11], jointList[3])
    drawLine(jointList[12], jointList[4])
    drawLine(jointList[7], jointList[3])
    drawLine(jointList[8], jointList[4])
    drawLine(jointList[7], jointList[8])
    """
    # headspin2
    # drawLine(jointList[11], jointList[12])
    # drawLine(jointList[11], jointList[3])
    # drawLine(jointList[12], jointList[4])
    # drawLine(jointList[7], jointList[3])
    # drawLine(jointList[7], jointList[4])
# art 7
def drawTriangles(jointList):
    drawTriangleFilled(jointList[8], jointList[9], jointList[12], (0,0,255))
    drawTriangleFilled2(jointList[9], jointList[10], jointList[11], (0,255,0))
    # drawTriangleFilled3(jointList[3], jointList[5], jointList[6], (0,255,0))
    #  drawTriangleFilled4(jointList[4], jointList[6], jointList[3], (0,0,255))

    # drawTriangleContour(jointList[9], jointList[10], jointList[11])

# scikit image
# under construction
def Traceaa(jointid):
    if results.pose_landmarks:
            
        for id, lm in enumerate(results.pose_landmarks.landmark):
            
            # opencv scikit image format conversion
            scikit_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, c = scikit_img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id == jointid:
                ptsTail1.appendleft((cx, cy))
                for i in range(1, len(ptsTail1)):
                    # if either of th etracked points are none, ignore them.
                    if ptsTail1[i-1] is None or ptsTail1[i] is None:
                        continue
                    # otherwise compute thickness of line n draw connecting lines
                    # thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
                    # cv2.line(image, ptsTail1[i-1], ptsTail1[i], (230, 230, 230), 2)
                    # image = img_as_float(image)  # convert to sci image
                    rr, cc, val = line_aa(ptsTail1[i-1][0], ptsTail1[i-1][1], ptsTail1[i][0], ptsTail1[i][1])                      
                    scikit_img[rr, cc] = val * 255
                    image = cv2.cvtColor(scikit_img, cv2.COLOR_RGB2BGR)
                    # image = img_as_ubyte(image)               
# art 9
def horizontalHalfLineLeft(pt, color):
    """
    pt: a tuple coordinate point
    """
    pt1 = pt
    pt2 = (0, pt[1])
    drawLine(pt1, pt2, color)

def horizontalHalfLineRight(pt, color):
    """
    pt: a tuple coordinate point
    """
    pt1 = pt
    pt2 = (width, pt[1])
    drawLine(pt1, pt2, color)

def verticalHalfLine(pt, color):

    pt1 = pt
    # pt2 = (pt[0], int(height/2))
    pt2 = (pt[0], 10)
    drawLine(pt1, pt2, color)

def verticalHalfLineFootwork(pt, color):
    pt1 = pt
    pt2 = (pt[0], int(height/2))
    drawLine(pt1, pt2, color)

def diagonalLine(pt):
    
    pt1 = pt
    pt2 = (width, height)
    drawLine(pt1, pt2)

def drawRectangle(pt):
    cv2.rectangle(image, (0,0), pt, color, -1)

def MaedaLines(jointidlist, color):
    """
    jointidlist : a list of joint IDs
    returns horizontal lines drawn at each 
    """
    for i in getCoordinatesList(jointidlist):
        # drawRectangle(i)
        # horizontalHalfLine(i)
        verticalHalfLine(i, color)
        # diagonalLine(i)

def MaedaLinesFootwork(jointidlist, color):
    """
    jointidlist : a list of joint IDs
    returns horizontal lines drawn at each 
    """
    for i in getCoordinatesList(jointidlist):
        # drawRectangle(i)
        # horizontalHalfLine(i)
        verticalHalfLineFootwork(i, color)
        # diagonalLine(i)

def MaedaLinesLeft(jointidlist, color):
    """
    jointidlist : a list of joint IDs
    returns horizontal lines drawn at each 
    """
    for i in getCoordinatesList(jointidlist):
        # drawRectangle(i)
        horizontalHalfLineLeft(i, color)
        # verticalHalfLine(i)
        # diagonalLine(i)

def MaedaLinesLeftRight(jointidlist, color):
    """
    jointidlist : a list of joint IDs
    returns horizontal lines drawn at each 
    """
    for i in getCoordinatesList(jointidlist):
        # drawRectangle(i)
        horizontalHalfLineRight(i, color)
        # verticalHalfLine(i)
        # diagonalLine(i)

# art 10
def waterPainting(image):
    # res = cv2.stylization(image, sigma_s=190, sigma_r=1.0)
    cv2.xphoto.oilPainting(image, 7, 1)

# art 11
def pencilSketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # invert = cv2.bitwise_not(gray)  # invert
    blur = cv2.GaussianBlur(gray, (25, 25), 15)
    # invertBlur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray, blur, scale = 256.0)

    return sketch

# art 12
# @widgets.interact_manual(s=(0,200,1),r=(0,1,0.1))
def edgePreserve(s=50,r=0.5):
    edgeImg = cv2.edgePreservingFilter(img,sigma_s=s,sigma_r=r)   

def opacityart(image):
    overlay = image.copy()

    # manipulate overlay
    drawTriangles(getCoordinatesList(jointids))
    # drawConsciousness()


    alpha = 0.7
    beta = 1-alpha
    cv2.addWeighted(overlay, alpha, image, beta, 0.0 , image)



if __name__ == "__main__":
    """
    if this file (a.k.a module) is being run directly 
    """

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6) as pose:
        # i mean for art we wish for inaccuracy, so just keep tracking.
        # it makes it faster also.
        # min track con 1 means it is starting detection again in each frame.
        # n then it fails some times for mindetcon = 0.5, list index out of range
      while cap.isOpened():
        success, image = cap.read()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 3
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 4

        fps = cv2.CAP_PROP_FPS           # 5
        frameCount = cv2.CAP_PROP_FRAME_COUNT   # 7


        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          break

        """
        # I think it is better to keep the getCoordinatesList() here because it
        # is needed everytime. Also for further image processing effect can be 
        # rendered afterwards because without being worried about if the 
        # processed image has to undergo POSE ESTIMATION algo !!
        """
        # image = img_as_float(image)  # cv2 image to sci image
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                  cv2.THRESH_BINARY, 199, 5)
   

    
        if args["black"]:
            image = image*0

        # drawRotoscopeBalletConnections(getCoordinatesList(jointids))
        # drawRotoscopeAllConnections(getCoordinatesList(jointids))
        # Body(getCoordinatesList(Art4jids))
        # keep one list with two Trace function calls... u se that zigzag pattern
        
        # drawTriangles(getCoordinatesList(jointids))
        # Traceaa(18)
        jointIDsForMaedaLines = [13, 14, 15, 16]
        jointIDsForMaedaLinesLeft = [26, 28, 32]
        jointIDsForMaedaLinesRight = [25, 27, 31]
        jointIDsverticalHalfLineFootwork = [25, 26, 27, 28, 31, 32]
        color = (200, 200, 200)
        

        # image = cv2.xphoto.oilPainting(image, 5, 1)
        # image = cv2.stylization(image, sigma_s=190, sigma_r=0.4)
        # image_gray, image_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)

        # drawConsciousness(jointid=[0,11,12])
        
        # image = pencilSketch(frame = image) 
        # MaedaLinesFootwork(jointIDsverticalHalfLineFootwork, color=(255, 255, 255))
        #MaedaLinesLeft(jointIDsForMaedaLinesLeft, color=(255, 255, 255))
        #MaedaLinesLeftRight(jointIDsForMaedaLinesRight, color=(255, 255, 255))
        #MaedaLines(jointIDsForMaedaLines, color=(255, 255, 255))
        
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(image, (25, 25), 15)
        # image = cv2.divide(gray, blur, scale = 256.0)
                
        # TraceTail1(21)
        # TraceTail2(23)
        # res = cv2.edgePreservingFilter(image,sigma_s=50,sigma_r=0.6)
        # res = cv2.detailEnhance(image,sigma_s=20,sigma_r=0.2)
        
        # parallalLines1(23)
        # parallalLines2(0)
        # waterPainting(image)
        
        opacityart(image)
        cv2.imshow('GrooveCanvas', image)
        out.write(image)
                        
        # go and read the next frame
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    out.release()

else:
    print("example module is imported into another module")

"""

double run:

draw the pencil sketch and then in the 2nd run draw on top of it.
what is happening is, image is different from processed ( gray, blur ) 
n that is why u really have to store the image itself, u can't store the
processed matrix .... is not it so. not sure check ha ha !!
"""
# https://opencv-tutorial.readthedocs.io/en/latest/draw/draw.html
# https://medium.com/analytics-vidhya/playing-with-images-using-opencv-5dca960d1b0b
