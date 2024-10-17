
* ໂປລເຈັກນີ້ແມ່ນ ຮຽນແລະຝິກນຳ ວີດີໂອຢູທູບ
***
https://www.youtube.com/watch?v=MJCSjXepaAM&t=181s

***
ໂດຍ libary ຄື :
***
*  mediapipe 0.9.0.1
 python 3.9"\
 scikit-learn 1.2.0

 ***
ຕອນທຳອິດ ໃນຄິອມຂອງຂ້ອຍໄດ້ ມີ scikit-learn  1.1.2 ຈຶງຕ້ອງໄດ້ ຕິດຕັ້ງເວີຊັິ່ນ 1.2.0 ໃຊ້ command :
*  $ pip install scikit-learn==1.2.0 $


 ພວກເຮົາໃຊ້ໄຟລ $file collect_imgs$ ເກັບດາຕ້າ
*  *ຫລັງຈາກເກັບຂຊ້ມູນແລ້ວໆ ເຮົາກະກວດເບິ່ງດາຕ້າ ຕາມ DATA_DIR ວ່າໂອເຄແລ້ວຫລືໍບໍ

****
ພວກເຮົາໃຊ້ໄຟລ $file collect_imgs$ ສ້າງດາຕ້າດາຕ້າ
 *  ຫລັງຈາກນັ້ນ ເຮົາທົດລອງເອົາອອກມາ
 *  results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )


****
ສຳລັບໂມເດວທີ່ໃຊ້ຝຶກແມ່ນໃຊ້ model RandomForestClassifier ໃນ scikit-learn ມາຊ່ວຍຝຶກ

***
and the evalution we use a accury_score
***
 
 