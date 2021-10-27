import numpy as np
import cv2
import time


frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)

# 320 416 608
whT = 608
confThreshold_detection = 0.8
nmsThreshold= 0.2

classesFiles = 'yolov3_mask_detection_tiny/classes.names'

with open(classesFiles, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3_mask_detection_tiny/yolov3_tiny_mask.cfg'
modelWeights = 'yolov3_mask_detection_tiny/yolov3_tiny_mask_final.weights'

#modelConfiguration = 'yolov3_mask_detection_tiny/yolov3_mask.cfg'
#modelWeights = 'yolov3_mask_detection_tiny/yolov3_mask_last.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def get_img_box():
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    result = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            class_name = classNames[classId]
            confidence = scores[classId]
            if confidence > confThreshold_detection:

                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                xx,yy = int(x + w/2) , int(y + h/2)

                bbox.append([x,y,w,h,xx,yy])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold_detection, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        label = classNames[classIds[i]]

        r = (bbox[i])
        result.append(r)

        fps = int(1.0 / (time.time() - start_time))
        

        if label == 'withmask':
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if label == 'masknotright':
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if label == 'withoutmask':
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        cv2.putText(img, "FPS: " + str(fps), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 0, 127), 2)
        
    return result


while True:
    success, frame_og = cap.read()
    frame_mirror = cv2.flip(frame_og,1)
    img = cv2.resize(frame_mirror, (frameWidth, frameHeight))

    start_time = time.time()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)

    net.setInput(blob)

    layersNames = net.getLayerNames()

    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    get_img_box()

    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()