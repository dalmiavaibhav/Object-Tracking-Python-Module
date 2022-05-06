import cv2
import ObjectTracking as oj

vs = cv2.VideoCapture('motor_bike.mp4')
_, init_frame = vs.read()
box = cv2.selectROI('frame', init_frame, showCrosshair=False)
tracker = oj.csrt.CSRT(init_frame, box)


#print(box)

while True:
    _, img = vs.read()

    tracks = tracker.track(img)
    
    #print(tracks)

    for t in tracks:
        cv2.rectangle(img, (t[0], t[1]), (t[2], t[3]), (0, 0, 255), 2, 1)
    cv2.imshow('frame', img)

    if cv2.waitKey(1) == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()