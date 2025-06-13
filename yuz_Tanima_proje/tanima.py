import cv2
import numpy as np
import os


tanıyıcı = cv2.face.LBPHFaceRecognizer_create()
veri_yolu = 'veri'
model_yolu = 'model.yml'


if not os.path.exists(model_yolu):
    resimler = []
    etiketler = []

    for dosya in os.listdir(veri_yolu):
        if dosya.startswith("User"):
            gri = cv2.imread(os.path.join(veri_yolu, dosya), cv2.IMREAD_GRAYSCALE)
            id_ = int(dosya.split('.')[1])
            resimler.append(gri)
            etiketler.append(id_)

    tanıyıcı.train(resimler, np.array(etiketler))
    tanıyıcı.save(model_yolu)
    print("Model eğitildi ve kaydedildi.")
else:
    tanıyıcı.read(model_yolu)


isimler = {
    1: "deneme",
    2: "halil",
    3: "ayse"
}

kamera = cv2.VideoCapture(0)
yuz_dedektor = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, kare = kamera.read()
    if not ret:
        break

    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    yuzler = yuz_dedektor.detectMultiScale(gri, 1.3, 5)

    for (x, y, w, h) in yuzler:
        yuz = gri[y:y+h, x:x+w]
        id_, dogruluk = tanıyıcı.predict(yuz)

        if dogruluk < 70:
            isim = isimler.get(id_, "Bilinmiyor")
        else:
            isim = "Bilinmiyor"

        cv2.rectangle(kare, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(kare, isim, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Yüz Tanima", kare)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
