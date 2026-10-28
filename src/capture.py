"""
Adım 1 — GStreamer → NumPy
Amaç: Kameradan frame al, NumPy array'e çevir, FPS ölç.
YOLOv8 henüz yok. Önce capture katmanının sağlıklı çalıştığını kanıtlıyoruz.
"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np
import cv2
import time

# GStreamer'ı başlat — her zaman ilk satır olmalı
Gst.init(None)


# ─────────────────────────────────────────
# 1. PIPELINE STRING
# ─────────────────────────────────────────
# Terminalde test ettiğimiz komutun aynısı,
# autovideosink → appsink olarak değişti.
# name=sink → Python'dan bu elementi bulmak için

PIPELINE_STR = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "queue max-size-buffers=2 leaky=downstream ! "   # ← yeni: buffer kuyruğu
    "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
)

# queue açıklaması:
#   max-size-buffers=2 → en fazla 2 frame beklesin
#   leaky=downstream   → kuyruk dolunca eski frame'i at, yenisini al
#   drop=true (appsink) → biz almadan yeni frame gelirse eskiyi sil
# Bu üç ayar birlikte "her zaman en güncel frame" garantisi verir.


# ─────────────────────────────────────────
# 2. YARDIMCI FONKSİYON: GstSample → NumPy
# ─────────────────────────────────────────
def gst_to_numpy(sample):
    """
    GStreamer'dan gelen ham sample'ı (1080, 1920, 3) boyutlu
    BGR NumPy array'e çevirir.
    """
    buf = sample.get_buffer()
    caps = sample.get_caps()

    # Çözünürlüğü caps'ten oku — hardcode etmiyoruz
    width  = caps.get_structure(0).get_value("width")
    height = caps.get_structure(0).get_value("height")

    # Ham byte'lara eriş
    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        return None

    try:
        # BGRx: 4 kanal (B, G, R, padding)
        frame = np.frombuffer(map_info.data, dtype=np.uint8)
        frame = frame.reshape((height, width, 4))
        frame = frame[:, :, :3]   # padding'i at → BGR
        return frame.copy()       # .copy(): buf serbest bırakılmadan önce veriyi kopyala
    finally:
        buf.unmap(map_info)       # finally: hata olsa da unmap çağrılsın


# ─────────────────────────────────────────
# 3. ANA SINIF
# ─────────────────────────────────────────
class CameraCapture:

    def __init__(self):
        self.pipeline = Gst.parse_launch(PIPELINE_STR)
        self.appsink  = self.pipeline.get_by_name("sink")

        if self.appsink is None:
            raise RuntimeError("appsink bulunamadı — pipeline string'ini kontrol et")

    def start(self):
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Pipeline PLAYING durumuna geçemedi")
        print("Pipeline başlatıldı")

    def read(self):
        """
        Bir sonraki frame'i döner.
        Frame yoksa None döner — döngü bunu kontrol etmeli.
        """
        sample = self.appsink.emit("pull-sample")
        if sample is None:
            return None
        return gst_to_numpy(sample)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        print("Pipeline durduruldu")


# ─────────────────────────────────────────
# 4. ÇALIŞTIR VE DOĞRULA
# ─────────────────────────────────────────
if __name__ == "__main__":

    cam = CameraCapture()
    cam.start()

    fps_counter = 0
    fps_display = 0.0
    t_start     = time.time()

    print("Çalışıyor — çıkmak için Ctrl+C")
    print("Her saniye FPS ve frame boyutu raporlanacak\n")

    try:
        while True:
            frame = cam.read()

            if frame is None:
                continue

            fps_counter += 1

            # Her saniye FPS hesapla
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                print(f"FPS: {fps_display:.1f}  |  Frame shape: {frame.shape}  |  dtype: {frame.dtype}")
                fps_counter = 0
                t_start     = time.time()

            # İsteğe bağlı: görüntüyü ekranda göster
            # (monitör yoksa bu satırı yoruma al)
            cv2.imshow("Capture Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nKullanıcı durdurdu")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
