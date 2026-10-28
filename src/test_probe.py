#!/usr/bin/env python3
import sys
sys.path.append('/home/hasan/deepstream_python_apps/apps')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.bus_call import bus_call
import pyds

MUXER_BATCH_TIMEOUT_USEC = 33000

def probe_callback(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

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

            rect = obj_meta.rect_params
            print(f"Frame:{frame_meta.frame_num} | "
                  f"ID:{obj_meta.object_id} | "
                  f"class:{obj_meta.class_id} | "
                  f"conf:{obj_meta.confidence:.2f} | "
                  f"left:{rect.left:.0f} top:{rect.top:.0f} "
                  f"w:{rect.width:.0f} h:{rect.height:.0f}")

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
    source.set_property("num-buffers", 300)

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

    tracker_src = tracker.get_static_pad("src")
    tracker_src.add_probe(Gst.PadProbeType.BUFFER, probe_callback, 0)
    print("Probe eklendi.")

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline basliyor...")

    try:
        loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)
    print("Pipeline durduruldu.")


if __name__ == "__main__":
    main()
