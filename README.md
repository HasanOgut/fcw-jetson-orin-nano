# Kenar Bilişim Üzerinde Gerçek Zamanlı İleri Çarpışma Uyarı Sistemi (FCW)

NVIDIA Jetson Orin Nano üzerinde, CSI kameradan gelen görüntüde yayaları tespit eden, her yayanın mesafesini ve çarpışmaya kalan süreyi (TTC) hesaplayıp **DİKKAT / TEHLİKE** uyarısı üreten uçtan uca bir sistem.

Kütahya Sağlık Bilimleri Üniversitesi, Bilgisayar Mühendisliği Lisans Bitirme Projesi (2026).

## Proje ne yapıyor?

Kamera önündeki yayayı bulur, ona bir kimlik verip takip eder, ne kadar uzakta olduğunu ve ne hızla yaklaştığını tahmin eder. Yaklaşma hızına göre çarpışmaya kaç saniye kaldığını hesaplar ve ekrana uyarı basar. Tüm işlem kenar cihazda, buluta görüntü göndermeden çalışır.

Projenin ikinci amacı, aynı sistemin içinde dokuz farklı nesne tespit modelini (PeopleNet, YOLOv8n/s/m, YOLOv9c, YOLOv10n, YOLO11n, YOLO26m, RT-DETR-l) FP32 ve FP16 hassasiyetinde karşılaştırmaktır.

## Sistem akışı

```mermaid
flowchart LR
    A["IMX477 CSI kamera<br/>1920x1080, 60 FPS"] --> B["nvarguscamerasrc<br/>(NVMM belleği, sıfır kopya)"]
    B --> C["nvstreammux<br/>(toplulaştırma)"]
    C --> D["nvinfer<br/>TensorRT motoru<br/>(nesne tespiti)"]
    D --> E["nvtracker<br/>NvDCF (takip kimliği)"]
    E --> F["Python pad probe<br/>mesafe + TTC hesabı"]
    F --> G["nvdsosd<br/>ekrana çizim"]
    G --> H["nv3dsink<br/>görüntü"]
```

Görüntü, kameradan ekrana kadar GPU belleğinde kalır. Python tarafı yalnızca nesne meta verisini (kutu, güven skoru, takip kimliği) okur, görüntüyü kopyalamaz.

## Mesafe ve TTC nasıl hesaplanıyor?

```mermaid
flowchart TD
    A["Takip edilen yaya<br/>(sınır kutusu yüksekliği h, piksel)"] --> B["Mesafe = gerçek boy x odak uzunluğu / h<br/>(1750 mm, 840 piksel)"]
    B --> C["Son 5 kareden mesafe geçmişi"]
    C --> D["Yaklaşma hızı =<br/>(önceki mesafe - şimdiki mesafe) / geçen süre"]
    D --> E{"Hız > 50 mm/s?"}
    E -- "Hayır (titreşim gürültüsü)" --> F["TTC hesaplanmaz"]
    E -- "Evet" --> G["TTC = mesafe / yaklaşma hızı"]
    G --> H{"TTC değeri"}
    H -- "TTC < 3 sn" --> I["TEHLİKE"]
    H -- "3 sn ile 5 sn arası" --> J["DİKKAT"]
    H -- "TTC > 5 sn" --> K["Uyarı yok"]
```

Mesafe, tek kamera ve sabit ortalama insan boyu varsayımıyla bulunan bir **mühendislik tahminidir**, kesin ölçüm değildir.

## Model hazırlama süreci

```mermaid
flowchart LR
    A["Eğitilmiş model<br/>(.pt / NGC)"] --> B["ONNX'e dönüştürme<br/>(DeepStream-Yolo export betikleri)"]
    B --> C["TensorRT motoru<br/>FP32 / FP16"]
    C --> D["İzole test<br/>trtexec (qps, gecikme)"]
    C --> E["Canlı test<br/>DeepStream (FPS)"]
    D --> F["Karşılaştırma"]
    E --> F
```

## Sonuçlardan örnekler

Jetson Orin Nano 8GB, DeepStream 7.1, CUDA 12.6.

**Canlı pipeline (FP32), kamera 60 FPS ile sınırlı:**

| Model | FPS |
|---|---|
| YOLO11n / YOLOv8n / YOLOv10n | ~59,7 - 59,9 (kamera sınırı) |
| YOLOv8s | 44,7 |
| PeopleNet | 28,9 |
| YOLOv8m | 24,7 |
| YOLOv9c | 20,0 |
| RT-DETR-l | 15,0 |

**İzole motor testi (trtexec), FP32 ve FP16 throughput:**

| Model | FP32 (qps) | FP16 (qps) | Kazanç |
|---|---|---|---|
| YOLOv10n | 94,9 | 208,9 | +%120 |
| YOLOv8s | 55,6 | 142,7 | +%157 |
| PeopleNet | 35,1 | 83,9 | +%139 |
| YOLOv8m | 29,8 | 68,4 | +%130 |
| YOLOv9c | 18,9 | 44,8 | +%138 |
| RT-DETR-l | 19,1 | 35,9 | +%88 |
| YOLO26m | 30,6 | 56,3 | +%84 |

Küçük modeller canlı pipeline'da kameranın 60 FPS sınırına takıldığı için FP16 kazancı yalnızca izole testte görülür. Yaya tespitine özel eğitilmiş PeopleNet modeli FP16 ile 47,8 FPS verdi ve test senaryosunda takip kimliğini en kararlı tutan seçeneklerden biri oldu.

## Karşılaşılan başlıca mühendislik sorunları

- Standart Ultralytics ONNX çıktısı, DeepStream-Yolo ayrıştırıcısıyla uyumsuz çıktı. Kütüphanenin kendi `export_*.py` betikleri kullanıldı.
- RT-DETR-l, farklı çıkış formatı nedeniyle tüm nesneleri "bisiklet" olarak etiketledi. `patches/nvdsparsebbox_Yolo.cpp` içinde özel bir ayrıştırıcı (`NvDsInferParseRTDETR`) yazılarak çözüldü.
- Sınır kutusu titreşimi anlamsız TTC değerleri üretiyordu. 50 mm/s altındaki yaklaşma hızları gürültü sayılarak elendi.

## Klasör yapısı

```
src/        Python betikleri (pad probe, TTC, benchmark)
configs/    DeepStream nvinfer, tracker (NvDCF) ve kaynak yapılandırmaları
patches/    DeepStream-Yolo'ya eklenen RT-DETR ayrıştırıcısı
```

Model, ONNX ve TensorRT motor dosyaları boyutları nedeniyle depoya dahil değildir.

## Kullanılan açık kaynak projeler

- [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo): YOLO ve RT-DETR modellerinin DeepStream'e aktarılması
- [deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps): Python (pyds) arayüzü
- [Ultralytics](https://docs.ultralytics.com/): YOLO modelleri
- [NVIDIA NGC PeopleNet](https://catalog.ngc.nvidia.com/): yaya tespitine özel model
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk), TensorRT, CUDA

## Sınırlılıklar

- Kontrollü iç mekân ortamında, tek yayalı senaryoyla test edildi.
- Yalnızca insan sınıfı için uyarı üretir.
- Mesafe tahmini tek kameraya ve sabit boy varsayımına dayanır.
- INT8 kuantizasyonu kapsam dışı bırakıldı.

## Yazar

Hasan Ögüt, Danışman: Dr. Öğr. Üyesi Ahmet Baştuğ

Proje raporu talep üzerine paylaşılabilir.
