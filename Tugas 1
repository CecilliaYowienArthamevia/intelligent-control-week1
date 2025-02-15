import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Rentang warna merah dalam HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Rentang warna hijau dalam HSV
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Rentang warna biru dalam HSV
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])

    # Masking untuk mendeteksi warna merah, hijau, dan biru
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Menggabungkan hasil mask untuk ketiga warna
    mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_green, mask_blue))

    # Menemukan kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Menambahkan bounding box dan keterangan warna
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Menghindari kontur kecil
            # Menentukan posisi bounding box
            x, y, w, h = cv2.boundingRect(contour)
            # Menambahkan bounding box pada frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Menambahkan keterangan warna
            if cv2.countNonZero(mask_red[y:y+h, x:x+w]) > 0:
                color = "Red"
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif cv2.countNonZero(mask_green[y:y+h, x:x+w]) > 0:
                color = "Green"
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif cv2.countNonZero(mask_blue[y:y+h, x:x+w]) > 0:
                color = "Blue"
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Menampilkan hasil di satu tampilan saja (frame dengan bounding box dan keterangan warna)
    cv2.imshow("Frame", frame)

    # Keluar dari program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break