#!/usr/bin/env python3
"""
DeepStream Pad Probe - Mesafe ve TTC Hesabi
IMX477-160: focal_length=840px, sensor 1920x1080
"""
import sys
sys.path.append('/home/hasan/deepstream_python_apps/apps')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.bus_call import bus_call
import pyds
import time

# Kamera parametreleri
FOCAL_LENGTH_PX = 840       # IMX477-160, 1920x1080 mod
REAL_PERSON_HEIGHT_MM = 1750  # ortalama insan boyu

# Tracker gecmisi: {object_id: [(timestamp, distance_mm), ...]}
track_history = {}

MUXER_BATCH_TIMEOUT_USEC = 33000


def estimate_distance(bbox_height_px):
    """Bounding box yuksekliginden mesafe tahmini (mm)"""
    if bbox_height_px <= 0:
        return None
    return (REAL_PERSON_HEIGHT_MM * FOCAL_LENGTH_PX) / bbox_height_px


def estimate_ttc(obj_id, current_dist_mm, current_time):
    """
    Time-to-Collision hesabi (saniye)
    Pozitif TTC: nesne yaklasiyor
    Negatif TTC: nesne uzaklisiyor
    """
    if obj_id not in track_history:
        track_history[obj_id] = []

    history = track_history[obj_id]
    history.append((current_time, current_dist_mm))

    # Son 5 frame'i tut
    if len(history) > 5:
        history.pop(0)

    if len(history) < 2:
        return None

    # Ortalama hiz hesabi (mm/s)
    dt = history[-1][0] - history[0][0]
    if dt <= 0:
        return None

    dd = history[0][1] - history[-1][1]  # pozitif = yaklasma
    velocity_mm_s = dd / dt

    if velocity_mm_s <= 0:
        return None  # uzaklisiyor veya sabit

    ttc = current_dist_mm / velocity_mm_s
    return ttc


def probe_callback(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    current_time = time.time()

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Sadece person (class 0)
            if obj_meta.class_id == 0:
                rect = obj_meta.rect_params
                bbox_h = rect.height
                obj_id = obj_meta.object_id

                dist_mm = estimate_distance(bbox_h)
                dist_m = dist_mm / 1000.0 if dist_mm else None

                ttc = estimate_ttc(obj_id, dist_mm, current_time) if dist_mm else None

                if dist_m is not None:
                    ttc_str = f"{ttc:.1f}s" if ttc else "---"
                    danger = ""
                    if ttc and ttc < 3.0:
                        danger = " *** TEHLIKE ***"
                    elif ttc and ttc < 5.0:
                        danger = " ! DIKKAT !"

                    print(f"Frame:{frame_meta.frame_num:5d} | "
                          f"ID:{obj_id:3d} | "
                          f"conf:{obj_meta.confidence:.2f} | "
                          f"bbox_h:{bbox_h:.0f}px | "
                          f"dist:{dist_m:.2f}m | "
                          f"TTC:{ttc_str}{danger}")

                # Temizlik - uzun sure gorulmeyen ID'leri sil
                active_ids = set()
                try:
                    active_ids.add(obj_id)
                except:
                    pass

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main():
    Gst.init(None)

    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make("nvarguscamerasrc", "src")

    caps1 = Gst.ElementFactory.make("capsfilter", "caps1")
    caps1.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1"))

    streammux = Gst.ElementFactory.make("nvstreammux", "mux")
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("live-source", 1)

    pgie = Gst.ElementFactory.make("nvinfer", "infer")
    pgie.set_property("config-file-path",
        "/home/hasan/deepstream_project/DeepStream-Yolo/config_infer_primary_yoloV8.txt")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file",
        "/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)

    osd = Gst.ElementFactory.make("nvdsosd", "osd")
    sink = Gst.ElementFactory.make("nv3dsink", "sink")
    sink.set_property("sync", 0)

    for el in [source, caps1, streammux, pgie, tracker, osd, sink]:
        pipeline.add(el)

    source.link(caps1)

    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps1.get_static_pad("src")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(osd)
    osd.link(sink)

    # Probe tracker cikisina ekle
    tracker_src = tracker.get_static_pad("src")
    tracker_src.add_probe(Gst.PadProbeType.BUFFER, probe_callback, 0)
    print("Probe eklendi. Pipeline basliyor...")
    print(f"Focal length: {FOCAL_LENGTH_PX}px | Insan boyu: {REAL_PERSON_HEIGHT_MM}mm")
    print("-" * 70)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)
    print("Pipeline durduruldu.")


if __name__ == "__main__":
    main()
