import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Rentang warna merah dalam HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    
    # Rentang warna biru dalam HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Rentang warna hijau dalam HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Gabungkan semua mask
    mask = mask_red + mask_blue + mask_green
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Temukan kontur untuk objek yang terdeteksi
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter area kecil agar tidak terdeteksi
            x, y, w, h = cv2.boundingRect(contour)
            color = (0, 255, 0)  # Default warna bounding box hijau
            label = "Unknown"
            
            # Tentukan warna yang terdeteksi berdasarkan mask
            if cv2.countNonZero(cv2.bitwise_and(mask_red, mask_red, mask=cv2.drawContours(np.zeros_like(mask_red), [contour], -1, 255, thickness=cv2.FILLED))) > 0:
                color = (0, 0, 255)  # Merah
                label = "Red"
            elif cv2.countNonZero(cv2.bitwise_and(mask_blue, mask_blue, mask=cv2.drawContours(np.zeros_like(mask_blue), [contour], -1, 255, thickness=cv2.FILLED))) > 0:
                color = (255, 0, 0)  # Biru
                label = "Blue"
            elif cv2.countNonZero(cv2.bitwise_and(mask_green, mask_green, mask=cv2.drawContours(np.zeros_like(mask_green), [contour], -1, 255, thickness=cv2.FILLED))) > 0:
                color = (0, 255, 0)  # Hijau
                label = "Green"
            
            # Gambar bounding box dengan warna yang sesuai
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Menampilkan hasil
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
