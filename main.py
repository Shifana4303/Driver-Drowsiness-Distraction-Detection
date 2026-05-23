import cv2
import numpy as np
from tensorflow.keras.models import load_model

eye_model = load_model('models/eye_model.h5')
yawn_model = load_model('models/yawn_model.h5')
dis_model = load_model('models/distraction_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (64,64))
    img = img / 255.0
    img = np.reshape(img, (1,64,64,3))

    eye_pred = eye_model.predict(img)
    yawn_pred = yawn_model.predict(img)
    dis_pred = dis_model.predict(img)

    if eye_pred < 0.5:
        cv2.putText(frame, "Eyes Closed!", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    if yawn_pred > 0.5:
        cv2.putText(frame, "Yawning!", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    if np.argmax(dis_pred) != 0:
        cv2.putText(frame, "Distracted!", (50,150),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Driver Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()