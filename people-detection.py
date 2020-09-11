# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
from imutils.video import VideoStream
from tempimage import TempImage
from imutils.video import FPS
import numpy as np
import argparse
import datetime
import imutils
import time
import cv2
import pyrebase
from multiprocessing import Process
import threading
import os
import moviepy.editor as moviepy


def convert_avi_to_mp4(avi_file_path, output_name):
    pro = os.popen(
        "ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict "
        "experimental -f mp4 '{output}'".format(
            input=avi_file_path, output=output_name))
    pro.read()
    return True


config = {
    "apiKey": "AIzaSyCNPKUOUeVNb8a6X4JYC2tth87AZFDfJ94",
    "authDomain": "test-video-622cd.firebaseapp.com",
    "databaseURL": "https://test-video-622cd.firebaseio.com",
    "projectId": "test-video-622cd",
    "storageBucket": "test-video-622cd.appspot.com",
    "messagingSenderId": "508557084861",
    "appId": "1:508557084861:web:d810ba3c6d7f9282d94fd4",
    "measurementId": "G-C4B00X54DT"
};
firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage = firebase.storage()


def push(cloud, local):
    storage.child(cloud).put(local)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    # ---------NEW-------------------------
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    # ------------------------------------
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # if the video argument is None, then we are reading from webcam
    if args.get("video", None) is None:
        #	vs = VideoStream(src=0).start()
        vs = cv2.VideoCapture(0)
        time.sleep(2.0)
        fps = FPS().start()

    # otherwise, we are reading from a video file
    else:
        vs = cv2.VideoCapture(args["video"])

    # initialize the first frame in the video stream
    firstFrame = None
    lastUploaded = datetime.datetime.now()
    motionCounter = 0
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))

    size = (frame_width, frame_height)

    # loop over the frames of the video

    m = 0
    p = False
    while True:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        ret, frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]
        text = "Unoccupied"
        detected = False
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            break

        """# resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)



        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        """
        # ------------NEW---------------------------
        frame = imutils.resize(frame, width=400)
        # print(frame)
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"] and int(detections[0, 0, i, 1]) == 15:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                detected = True
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if m == 10:
                    # Store image
                    timestamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
                    date = datetime.datetime.now().strftime("%d-%b-%Y")
                    time = datetime.datetime.now().strftime("%H:%M:%S")
                    # t = TempImage()
                    image_name = '{ts}.jpg'.format(ts=time)
                    cv2.imwrite(image_name, frame)
                    image_cloud_path = 'images/{date}/{img}'.format(date=date, img=image_name)
                    storage.child(image_cloud_path).put(image_name)
                    imgRef = storage.child('images/{date}/{img}'.format(date=date, img=image_name)).get_url(None)
                    db.child("test1").child(date).child(time).update({"name": time})
                    db.child("test1").child(date).child(time).update({"search": time})
                    db.child("test1").child(date).child(time).update({"image": imgRef})
                    # Store video
                    video_name_avi = "({ts}).avi".format(ts=time)
                    video_name_mp4 = "({ts}).mp4".format(ts=time)
                    result = cv2.VideoWriter(video_name_avi, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
                    video_cloud_path = 'videos/{date}/{vid}'.format(date=date, vid=video_name_mp4)
                    print("begin to record")
                    m += 1
                elif m > 10:
                    ret, frame1 = vs.read()
                    if m > 110:
                        m = 0
                        print("end recording")
                        p = True
                        # storage.child(video_cloud_path).put(video_name)
                        vidRef = storage.child('videos/{date}/{vid}'.format(date=date, vid=video_name_mp4)).get_url(
                            None)
                        db.child("test1").child(date).child(time).update({"video": vidRef})
                    else:
                        m += 1
                    if ret is True:
                        result.write(frame1)
                else:
                    m += 1
                # draw the prediction on the frame
                # label = "{}: {:.2f}%".format(CLASSES[idx],
                #                             confidence * 100)
                # cv2.rectangle(frame, (startX, startY), (endX, endY),
                #              COLORS[idx], 2)
                # y = startY - 15 if startY - 15 > 15 else startY + 15
                # cv2.putText(frame, label, (startX, y),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        if detected is False:
            if m > 10:
                # storage.child(video_cloud_path).put(video_name)
                p = True
                vidRef = storage.child('videos/{date}/{vid}'.format(date=date, vid=video_name_mp4)).get_url(None)
                db.child("test1").child(date).child(time).update({"video": vidRef})
                m = 0
            else:
                m = 0
        print("m = ", m)
        if p is True:
            p = False
            video_name_avi_path = "/home/tuan/Downloads/people-detection/{vid}".format(vid=video_name_avi)
            convert_avi_to_mp4(video_name_avi_path, video_name_mp4)
            print("p is True")
            print(os.path.getsize(video_name_mp4))
            storage.child(video_cloud_path).put(video_name_mp4)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # print(os.path.getsize(video_name))
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

        # ----------------------------------
        """
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if text == "Occupied":
            if detected == False:
                lastSeen = datetime.datetime.now()
                print(lastSeen)
                detected = True
                print(detected)
                timestamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
                name = '{ts}.avi'.format(ts=timestamp)
                result = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
            else:			
                ret, frame1 = vs.read()
                if ret == True:
                    cv2.imshow('Frame', frame1)
                    result.write(frame1)
                    if cv2.waitKey(1) & 0xFF == ord('q'): 
                            break

                # increment the motion counter
                motionCounter += 1
                print("Motion")
                print(motionCounter)

                # check to see if the number of frames with consistent motion is
                # high enough
                #if motionCounter >= 20:
                #	t = TempImage()
                #	cv2.imwrite(t.path, frame)
                #	lastUploaded = timestamp
                #	motionCounter = 0
        else :
            detected = False
            print(detected)
        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
        """

    # cleanup the camera and close any open windows
    fps.stop()
    vs.release()
    # if args.get("video", None) is None else vs.release()
    cv2.destroyAllWindows()
