import cv2
import time

cap = cv2.VideoCapture(0)  # video capture source camera (Here webcam of laptop)
ret, frame = cap.read()  # return a single frame in variable `frame`

time.sleep(0.5)

cv2.imwrite('c1.png', frame)
cv2.destroyAllWindows()

cap.release()