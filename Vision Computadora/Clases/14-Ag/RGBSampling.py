import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

x = 0;

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
    # if frame is read correctly ret is True
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")

gray2 = frame[:,:,0]
gray2.astype(float)

while True:
    x=x+1;
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    cv.imshow('Color', frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray.astype(float)
    grayN = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    # Display the resulting frame
    cv.imshow('Gray', grayN)

    b,g,r = cv.split(frame)
    img = cv.merge((b,r,g))
    cv.imshow('Martian', img)
    resample = cv.resize(gray, (0, 0), fx = 0.25, fy = 0.25,interpolation = cv.INTER_NEAREST)
    cv.imshow('Sampled', resample)
    
    resample2 = cv.resize(resample, (0, 0), fx = 4.0, fy = 4.0,interpolation = cv.INTER_NEAREST)
    cv.imshow('Sam_Zoom', resample2)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
