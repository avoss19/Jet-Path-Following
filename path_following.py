#!/usr/bin/python
import cv2 as cv
from settings import *
from math import cos, sin
from os import path

from bsmLib.misc import map
from math import pi
from adafruit_servokit import ServoKit

from time import time, sleep

def scaleView(src, scale = IMG_SCALE, viewName = "view"):
    src = cv.resize(src, (0, 0), fx=IMG_SCALE, fy=IMG_SCALE) # scale images
    cv.imshow(viewName, src)

def videoSource(src = 0, xRes = 1920, yRes = 1080, fps = 60):
    log.debug("Creating video src")

    vidSrc = cv.VideoCapture(src)

    # Set camera resolution
    if(type(src) == int):
       log.debug("Setting camera config")

       vidSrc.set(3, xRes)
       vidSrc.set(4, yRes)

       vidSrc.set(15, fps)

    X_RES = vidSrc.get(cv.CAP_PROP_FRAME_WIDTH)
    Y_RES = vidSrc.get(cv.CAP_PROP_FRAME_HEIGHT)
    FPS = vidSrc.get(cv.CAP_PROP_FPS)
    CODEC = vidSrc.get(cv.CAP_PROP_FOURCC)

    log.debug("Camera XRES: %d\tYRES: %d\tFPS: %d\tCODEC: %s" % (X_RES, Y_RES, FPS, hex(int(CODEC))))

    return vidSrc

def load_yolo():
    net = cv.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

    if(CUDA):
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def detect_objects(img, net, outputLayers):
    blob = cv.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            # print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):
	indexes = cv.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv.putText(img, label, (x, y - 5), font, 1, color, 1)
	# cv.imshow("Image", img)

def videoView(src, edges, draw, hsv, hsv_edges, hsv_draw):
    # Display Source Img, Edges, Draw
    rows_rgb, cols_rgb, channels = src.shape
    rows_gray, cols_gray = edges.shape

    rows_comb = max(rows_rgb, rows_gray)
    cols_comb = cols_rgb + cols_gray
    comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)

    comb[:rows_rgb, :cols_rgb] = src
    comb[:rows_gray, cols_rgb:] = edges[:, :, None]

    imageView = np.concatenate((comb, draw), axis=1)

    # Display Source Img, Edges, Draw
    rows_rgb, cols_rgb, channels = hsv.shape
    rows_gray, cols_gray = hsv_edges.shape

    rows_comb = max(rows_rgb, rows_gray)
    cols_comb = cols_rgb + cols_gray
    comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)

    comb[:rows_rgb, :cols_rgb] = hsv
    comb[:rows_gray, cols_rgb:] = hsv_edges[:, :, None]

    imageViewRow1 = np.concatenate((comb, hsv_draw), axis=1)

    imageView = np.concatenate((imageView, imageViewRow1), axis=0)

    imageView = cv.resize(imageView, (0, 0), fx=IMG_SCALE, fy=IMG_SCALE) # scale images

    # cv.imshow("imageView", imageView)

def videoViewAlt(src, edges, draw):
    # Display Source Img, Edges, Draw
    rows_rgb, cols_rgb, channels = src.shape
    rows_gray, cols_gray = edges.shape

    rows_comb = rows_rgb + rows_gray
    cols_comb = max(cols_rgb, cols_gray)
    comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)
    
    comb[:rows_rgb, :] = src
    comb[rows_rgb:, :] = edges[:, :, None]

    imageView = np.concatenate((comb, draw), axis=0)

    imageView = cv.resize(imageView, (0, 0), fx=IMG_SCALE, fy=IMG_SCALE) # scale images

    cv.imshow("imageVqiew", imageView)

def edgeDetectionHSV(src, blur = (15, 15)):
    blur = cv.blur(src, blur)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HLS)

    red = hsv[:, :, 0]
    green = hsv[:, :, 1]
    blue = hsv[:, :, 2]

    tmp = np.zeros_like(src)
    tmp[:, :, 1] = green
    tmp[:, :, 2] = blue
    hsv = tmp

    # cv.imshow("red", red)
    # cv.imshow("green", green)
    # cv.imshow("blue", blue)
    #
    # cv.imshow("hsv", hsv)


    kernel = np.ones((20, 20), np.uint8)
    mask = cv.inRange(hsv, LOWER_MASK_COLOR, UPPER_MASK_COLOR)
    # cv.imshow("colorMask", mask)
    # mask = cv.dilate(mask, kernel, iterations = 1)
    # cv.imshow("dilateColorMask", mask)

    edges = cv.Canny(mask, 50, 120)

    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contoursDraw = np.zeros_like(src)

    lines = cv.HoughLinesP(edges, 1, np.pi/180, THRESH, maxLineGap=MAX_LINE_GAP, minLineLength=MIN_LINE_LENGTH)

    return edges, lines

def edgeDetectionHLS(src, blurParam = (20, 20)):
    # blur = cv.blur(src, blurParam)

    # guassian = cv.GaussianBlur(blur, (5,5),0)
    # guassian = cv.GaussianBlur(guassian, (5,5),0)
    # guassian = cv.GaussianBlur(guassian, (5,5),0)

    hls = cv.cvtColor(src, cv.COLOR_BGR2HLS)

    red = hls[:, :, 0]
    green = hls[:, :, 1]
    blue = hls[:, :, 2]

    green = cv.inRange(green, np.array([160, 160, 160]), np.array([200, 200, 200]))

    bilateral = cv.bilateralFilter(red, 11, 17, 17)

    edges = cv.Canny(bilateral, 35, 80)

    lines = cv.HoughLinesP(edges, 1, np.pi/180, THRESH, maxLineGap=MAX_LINE_GAP, minLineLength=MIN_LINE_LENGTH)

    return edges, lines

def findContours(edges):
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:int(len(contours)*LINE_RANK)]
    return contours

def drawContours(src, contours):
    if contours is not None:
        for i, cont in enumerate(contours):
            epsilon = 0.001 * cv.arcLength(cont, True)
            approximations = cv.approxPolyDP(cont, epsilon, True)
            # cv.drawContours(src, [approximations], 0, (0,0,255), 10)
            cv.drawContours(src, [approximations], 0, (255,255,255), thickness=cv.FILLED)
            # cv.drawContours(src, cont, -1, (0,0,255), 10)

def drawLines(src, lines, color = (255, 0, 0), thickness = 20):
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]

            cv.line(src, (x1, y1), (x2, y2), color, thickness)

def warpView(src):
    transformMatrix = cv.getPerspectiveTransform(ROI, RES)
    warp = cv.warpPerspective(src, transformMatrix, (X_RES, Y_RES), flags=(cv.INTER_LINEAR))
    # cv.imshow("warp", warp)

    return warp

def drawROI(src):
    cv.line(src, np.int_(ROI[0]), np.int_(ROI[1]), (0, 0, 255), 10) # Left
    cv.line(src, np.int_(ROI[3]), np.int_(ROI[2]), (0, 0, 255), 10) # Right
    cv.line(src, np.int_(ROI[1]), np.int_(ROI[2]), (0, 0, 255), 10) # Top
    cv.line(src, np.int_(ROI[0]), np.int_(ROI[3]), (0, 0, 255), 10) # Bottom

# def bearing(edges, lines, maxBaseDif = 10):
def bearing(lines):
    p0 = lines[:,0,0:2]
    p1 = lines[:,0,2:4]

    thetas = np.arctan2(p1[:,1] - p0[:,1], p1[:,0] - p0[:,0])
    thetas[thetas < 0] = thetas[thetas < 0] + np.pi

    thetaAvg = np.sum(thetas) / len(thetas)

    return thetaAvg

def drawBearing(src, bearing):
    x0 = int(X_RES / 2)
    y0 = int(Y_RES)
    # print(bearing)
    x1 = int(-(cos(bearing) * 600) + x0)
    y1 = int(-(sin(bearing) * 600) + y0)
    # print("p0: (%d, %d)\tp1: (%d, %d)" % (x0, y0, x1, y1))
    cv.line(src, (x0, y0), (x1, y1), (0, 0, 255), 20)

def drive(kit, throttle, bearing):
    bearing = bearing * 180 / pi

    if(bearing > 120):
        bearing = 120
    elif(bearing < 60):
        bearing = 60

    throttle = map(throttle, -1, 1, MIDPOINT - MAX_THROTTLE, MIDPOINT + MAX_THROTTLE)

    kit.servo[SERVO].angle = bearing
    kit.servo[MOTOR].angle = throttle

    return throttle, bearing

def stop(kit):
    kit.servo[SERVO].angle = MIDPOINT
    kit.servo[MOTOR].angle = MIDPOINT

def main():
    cam = videoSource(SOURCE, X_RES, Y_RES, FPS)

    kit = ServoKit(channels=16)

    stop(kit)

    model, classes, colors, output_layers = load_yolo()

    theta = np.pi

    if(RECORD):
        i = 0
        file = RECORD_FILE % (i)

        log.debug("Starting recording: %s" % (file))

        while(path.isfile(file)):
            i = i + 1
            file = RECORD_FILE % (i)
            log.warning("File %s exist, renaming to %s" % (RECORD_FILE % (i - 1), file))

        rec = cv.VideoWriter(file, cv.VideoWriter_fourcc(*'MJPG'), 10, (X_RES, Y_RES))

    sum = 0
    frameNum = 0
    while(True):
        now = time()
        ret, frame = cam.read()

        # Reload source
        if(not ret):
            log.warning("Restarting video source")
            cam = videoSource(SOURCE, X_RES, Y_RES, FPS)
            sum = 0
            continue
        elif(RECORD):
            rec.write(frame)

        if(type(SOURCE) == int):
            frame = cv.undistort(frame, MTX, DIST, None, MTX)

        draw = frame.copy()

        height, width, channels = draw.shape
        blob, outputs = detect_objects(draw, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, draw)

        # mask, roi = findROI(frame, draw)
        warp = warpView(frame)
        edges, lines = edgeDetectionHSV(warp)

        # contours = findContours(edges)

        if lines is not None:
            theta = bearing(lines)

        drawLines(draw, lines)

        # contoursDraw = frame.copy()
        # drawContours(contoursDraw, contours)

        bearingDraw = frame.copy()
        drawBearing(draw, theta)
        drawROI(frame)
        videoViewAlt(frame, edges, draw)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        t = time() - now
        rate = 1/t
        sum += rate
        frameNum += 1

        throttle = 1
        if(len(boxes) > 0):
            throttle = 0
        drive(kit, throttle, theta)

        print("Time: %f\tFPS: %f\tAverage: %f\tDrive: %f %f" % (t, rate, sum/frameNum, throttle, theta))

    log.debug("Releasing recording & camera")
    stop(kit)
    if(RECORD):
        log.debug("Releasing recoding")
        rec.release()
    log.debug("Releasing video source")
    cam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
