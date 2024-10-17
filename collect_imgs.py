import os
import cv2


DATA_DIR = './langgue_Detect\data'
# ຖ້າບໍມີ ໄຟລ data ໃຫ້ສ້າງຂື້ນມາ
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
 
 
# ຮຸບແບ່ງງອອກເປັນ5ຈຳພວກ ແຕ່ລະພວກມີ 100 ຮຸບ    
number_of_classes = 5
dataset_size = 100


cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))
        
    print('Collecting data for class {}'.format(j))
    
    done = False
    while True:
        ret,frame = cap.read()
        cv2.putText(frame,'Ready ？press "L" !  :]',(100,50),cv2.FONT_ITALIC,1.3,
                    (0,196,255),3,cv2.LINE_AA)
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) == ord('l'):
            break
        
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg '.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
        