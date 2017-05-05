#!/usr/bin/env python

'''
Hand raising detection

USAGE:
    course_correction.py [--face-cascade <cascade_fn>] [--hand-cascade <cascade_fn>] [<video_source>]
'''

# hand detection added on top of OpenCV face detection sample using haar cascades

# Hand Haar cascades from
# https://github.com/Balaje/OpenCV/blob/master/haarcascades/palm.xml

# Hand raising detection loosely inspired by
# http://conteudo.icmc.usp.br/pessoas/moacir/papers/NazarePonti_CIARP2013.pdf

# Aruco inspiration from
# https://gist.github.com/hauptmech/6b8ca2c05a3d935c97b1c75ec9ad85ff

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str

# firebase

import pyrebase

config = {
  "apiKey": "",
  "authDomain": "",
  "databaseURL": "",
  "projectId": "",
  "storageBucket": "",
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()
users = db.child("users").get()
users = list(filter(lambda u: u != None, list(map(lambda u: u.val(), users.each()))))

hand_raise_queue = {}

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def marker_center(marker):
    q = marker[0]
    return [int((q[0][0] + q[1][0] + q[2][0] + q[3][0])/4), int((q[0][1] + q[1][1] + q[2][1] + q[3][1])/4)]

def user_with_marker(marker):
    return next((x for x in users if x['markerId'] == marker), None)

hand_raise_buffer_size = 12

def track_hand_raise(person, hand_raised):
    t = clock()
    #print(person)
    userId = person["id"]
    entry = hand_raise_queue.get(userId)
    if entry == None:
        entry = {"user": person, "send_t": 0, "events": []}
        hand_raise_queue[userId] = entry
    events = entry["events"]
    events.insert(0, hand_raised)
    while len(events) > hand_raise_buffer_size:
        events.pop()

def send_hand_raises(queue):
    # for now, only worry about hand raises, not lowering of hands
    threshold_raises = 4
    send_threshold_t = 20
    send_queue = {k: v for k, v in queue.items() if v["events"].count(True) > threshold_raises and clock() - v["send_t"] > send_threshold_t}
    for key, entry in send_queue.items():
        user = entry["user"]
        print('%s raised hand!' % user["name"])
        entry["send_t"] = clock()

def filter_rects_by_min_height(rects, min_height):
    return list(filter(lambda r: r[3] - r[1] > min_height, rects))

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['face-cascade=', 'hand-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    face_fn = args.get('--face-cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    hand_fn = args.get('--hand-cascade', "data/haarcascades/palm.xml")

    face_cascade = cv2.CascadeClassifier(face_fn)
    hand_cascade = cv2.CascadeClassifier(hand_fn)

    ar_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    while True:
        t = clock()

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        cam_dt = clock() - t
        t = clock()

        rects = detect(gray, face_cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))

        face_det_dt = clock() - t
        t = clock()

        markers = cv2.aruco.detectMarkers(gray, ar_dictionary)

        # Find centers of the detected markers
        centers = list(map(lambda q: marker_center(q), markers[0]))
        marker_ids = markers[1] # list(map(lambda q: people[q]["name"], markers[1]))
        markers_found = len(markers[0]) > 0
        if markers_found:
            #print('Detected!', len(markers))
            #print(markers[0],markers[1],len(markers[2]))
            cv2.aruco.drawDetectedMarkers(vis,markers[0],markers[1])
            #print('Centers:')
            #print(centers)
            #print('Names:')
            #print(marker_ids)

        marker_det_dt = clock() - t
        t = clock()

        # Detect hands to left/right of faces
        for p in rects:
            hand_raised = False
            x1, y1, x2, y2 = p
            face_w = x2 - x1
            face_h = y2 - y1
            left_x1 = max(x1 - face_w*2, 0)
            left_x2 = x1
            right_x1 = x2
            right_x2 = min(x2 + face_w*2, gray.shape[1])

            new_y1 = max(y1 - face_h*2, 0)
            new_y2 = y2

            x1, y1, x2, y2 = left_x1, new_y1, left_x2, new_y2
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            subrects = detect(roi.copy(), hand_cascade)
            subrects = filter_rects_by_min_height(subrects, face_h*0.4)
            draw_rects(vis_roi, subrects, (255, 0, 0))
            if len(subrects) > 0:
                hand_raised = True

            #print('right_x1: %.1f, right_x2: %.1f' % (right_x1, right_x2))
            x1, y1, x2, y2 = right_x1, new_y1, right_x2, new_y2
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            subrects = detect(roi.copy(), hand_cascade)
            subrects = filter_rects_by_min_height(subrects, face_h*0.4)
            draw_rects(vis_roi, subrects, (255, 0, 0))
            if len(subrects) > 0:
                hand_raised = True

            x1, y1, x2, y2 = p

            if hand_raised:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 10)
                draw_str(vis, (x1, y2+20), 'Hand is raised!')

            if markers_found:
                c = (int((x1 + x2)/2), int((y1 + y2)/2))
                distances = list(map(lambda q: (q[0] - c[0])**2 + (q[1] - c[1])**2, centers))
                val, closest_idx = min((val, idx) for (idx, val) in enumerate(distances))
                #print('Closest index: %d, name: %d' % (closest_idx, marker_ids[closest_idx]))
                closest_marker = int(marker_ids[closest_idx])
                #draw_str(vis, c, 'Marker %d' % closest_marker)
                person = user_with_marker(closest_marker)
                if person != None:
                    draw_str(vis, (x1, y2), person["name"])
                    track_hand_raise(person, hand_raised)

            # Match markers and faces
            #face_centers = list(map(lambda q: [int((q[0] + q[2])/2), int((q[1] + q[3])/2)], rects))
            #if len(face_centers) > 0:
            #    for p in centers:
            #        cv2.line(vis, (face_centers[0][0], face_centers[0][1]), (p[0], p[1]), (255, 0, 0))


        matching_dt = clock() - t

        send_hand_raises(hand_raise_queue)

        draw_str(vis, (20, 20), 'time cam: %.1f ms, face: %.1f ms, marker: %.1f, match: %.1f' % (cam_dt*1000, face_det_dt*1000, marker_det_dt*1000, matching_dt*1000))

        cv2.imshow('facedetect', vis)

        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()

