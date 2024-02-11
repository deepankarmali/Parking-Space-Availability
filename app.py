import streamlit as st
import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time

cap = cv2.VideoCapture(0)

data_dict = {}

total_parking_spots = 50

st.title("Parking System")

camera_feed_placeholder = st.empty()
info_placeholder = st.empty()

last_available_spots = None

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error reading frame. Exiting...")
        break

    frame = imutils.resize(frame, width=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 170, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

    if NumberPlateCnt is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        x, y, w, h = cv2.boundingRect(NumberPlateCnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        camera_feed_placeholder.image(frame, channels="BGR", caption="Camera Feed", use_column_width=True)

        config = ('-l eng --oem 1 --psm 3')

        text = pytesseract.image_to_string(new_image, config=config)

        if text.strip():
            if text in data_dict:
                current_time = time.time()
                last_entry_time = data_dict[text]
                if current_time - last_entry_time > 30:
                    st.success(f"Parking Spot Released for License Plate: {text}")
                    data_dict.pop(text) 
                else:
                    st.warning(f"License Plate detected again within 30 seconds: {text}")
            else:
                if len(data_dict) < total_parking_spots:
                    entry_time = time.time()
                    data_dict[text] = entry_time
                    st.success("Parking Successful!")

                    with open('parking_data.csv', 'a') as f:
                        f.write(f"{text},{entry_time}\n")

                else:
                    st.error("No available parking spots.")
            

    available_spots = total_parking_spots - len(data_dict)
    if last_available_spots is None or available_spots != last_available_spots:
        info_placeholder.info(f"Available Parking Spots: {available_spots}/{total_parking_spots}")
        last_available_spots = available_spots

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
