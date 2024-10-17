import pickle
from  PIL import ImageFont,Image,ImageDraw
import mediapipe as mp
import numpy as np

import cv2 

model_dict = pickle.load(open('./langgue_Detect\model.p','rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode =True,min_detection_confidence=0.3)

labels_dict = {0:'ຂ້ອຍ',1:'ເຮ້ !',2:'ຮັກ',3:'ເຈົ້າ',4:'ເດີ້ !'}
font_path = "./langgue_Detect\Deng.ttf"
font = ImageFont.truetype(font_path, 32)

while True:
    data_aux=[]
    
    x_ =[]
    y_ =[]
    ret,frame = cap.read()
    
    H,W, _ =frame.shape
    # cv2.putText(frame,' R      !!     L ',(100,50),cv2.FONT_HERSHEY_COMPLEX,1.3,
                    # (0,196,255),3,cv2.LINE_AA)
    
    b,g,r,a = 0,255,0,0
    
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    draw.text((200, 50),  "我             你", font = font, fill = (b, g, r))
    frame = np.array(frame_pil)
        
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
        
        
        x1 = int(min(x_)*W)
        y1 = int(min(y_)*H)
        x2 = int(max(x_)*W)
        y2 = int(max(y_)*H)
        
        prediction =model.predict([np.asarray(data_aux)])            
                    
        predicted_character = labels_dict[int(prediction[0])]
        
        cv2.rectangle(frame,(x1-20,y1-20),(x2+20,y2+20),(0,255,0),4)
        # cv2.putText(frame,predicted_character,(x1-22,y1-22),cv2.FONT_ITALIC,1.3,
        #                 (128,0,255),4,cv2.LINE_AA)   
        
        
        frame_pil1 = Image.fromarray(frame)
        draw1 = ImageDraw.Draw(frame_pil1)
        draw1.text((x1-22,y1-22),predicted_character , font = font, fill = (0, 224, 255))
        frame = np.array(frame_pil1)
            
    cv2.imshow('frame',frame)
    cv2.waitKey(25)

    
cap.release()
cv2.destroyAllWindows()