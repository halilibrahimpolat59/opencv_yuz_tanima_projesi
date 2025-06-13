import cv2
import os



kamera = cv2.VideoCapture(0)


yuz_dedektor = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


kullanici_id = 2 #hocam yeni veri kaydetinizde id numarasini bir artirin
sayi = 0


if not os.path.exists('veri'):
    os.makedirs('veri')

while True:
    ret, resim = kamera.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    yuzler = yuz_dedektor.detectMultiScale(gri, 1.3, 5)

    for (x, y, w, h) in yuzler:
        sayi += 1
        yuz_kirp = gri[y:y+h, x:x+w]
        cv2.imwrite(f"veri/User.{kullanici_id}.{sayi}.jpg", yuz_kirp)
        cv2.rectangle(resim, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Yuz Verisi Toplama", resim)

    if cv2.waitKey(20) & 0xFF == ord('q') or sayi >= 50:
        break

kamera.release()
cv2.destroyAllWindows()
