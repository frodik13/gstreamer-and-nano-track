mod trackers;
mod utils;
mod yolo;

use gstreamer::prelude::*;
use gstreamer::{Pipeline};
use opencv::prelude::*;
use opencv::{core, imgproc};
use opencv::core::{Rect, Scalar};
use crate::trackers::NanoTrack;
use crate::utils::{iou, mat_to_ndarray};
use crate::yolo::YoloV8;

fn main() -> opencv::Result<()> {
    gstreamer::init().unwrap();

    let pipeline_in_str = concat!(
        "libcamerasrc ! ",
        "videoconvert ! ",
        "video/x-raw,format=RGB,width=1632,height=1232,framerate=10/1 ! ",
        "appsink name=sink sync=false max-buffers=1 drop=true"
    );

    let pipeline_in = gstreamer::parse::launch(pipeline_in_str).expect("Can't launch pipeline");
    let pipeline_in = pipeline_in
        .dynamic_cast::<Pipeline>()
        .expect("Couldn't cast pipeline to pipeline");

    let appsink = pipeline_in
        .by_name("sink")
        .unwrap()
        .dynamic_cast::<gstreamer_app::AppSink>()
        .unwrap();

    let pipeline_out = gstreamer::parse::launch(
        "appsrc name=src is-live=true block=true format=time ! videoconvert ! kmssink",
    )
    .unwrap()
    .dynamic_cast::<Pipeline>()
    .unwrap();

    let appsrc = pipeline_out
        .by_name("src")
        .unwrap()
        .dynamic_cast::<gstreamer_app::AppSrc>()
        .unwrap();

    let width = 1632i32;
    let height = 1232i32;
    appsrc.set_caps(Some(
        &gstreamer::Caps::builder("video/x-raw")
            .field("format", &"RGB")
            .field("width", &width)
            .field("height", &height)
            .field("framerate", &gstreamer::Fraction::new(10, 1))
            .build(),
    ));

    pipeline_in
        .set_state(gstreamer::State::Playing)
        .expect("Can't set pipeline out");

    pipeline_out
        .set_state(gstreamer::State::Playing)
        .expect("Can't set pipeline out");

    {
        let bus = pipeline_in.bus().unwrap();
        std::thread::spawn(move || {
            for msg in bus.iter_timed(gstreamer::ClockTime::NONE) {
                use gstreamer::MessageView;
                match msg.view() {
                    MessageView::Error(err) => {
                        eprintln!(
                            "Pipeline (in) error from {:?}: {} ({:?})",
                            err.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        break;
                    }
                    MessageView::Eos(_) => {
                        eprintln!("Pipeline (in) EOS");
                        break;
                    }
                    _ => {}
                }
            }
        });
    }

    let appsink_thread = appsink.clone();
    let appsrc_thread = appsrc.clone();

    std::thread::spawn(move || {
        let mut yolo = YoloV8::new().unwrap();
        let mut nano_track: Option<NanoTrack> = None;
        let mut last_bbox: Option<Rect> = None;
        loop {
            match appsink_thread.try_pull_sample(gstreamer::ClockTime::from_seconds(5)) {
                None => {
                    println!("Can't pull sample");
                }
                Some(sample) => {
                    let buffer = match sample.buffer() {
                        None => {
                            eprintln!("Can't get buffer");
                            continue;
                        }
                        Some(b) => b,
                    };

                    let caps = sample.caps().expect("Can't get caps");
                    let s = caps.structure(0).expect("Can't get structure");
                    let w = s.get::<i32>("width").expect("Can't get width");
                    let h = s.get::<i32>("height").expect("Can't get height");

                    let map = match buffer.map_readable() {
                        Ok(m) => m,
                        Err(err) => {
                            eprintln!("Can't get map: {}", err);
                            continue;
                        }
                    };

                    let mut data = map.as_slice().to_vec();

                    let mut mat = match Mat::new_rows_cols_with_bytes_mut::<u8>(h, w * 3, &mut data)
                    {
                        Ok(m) => m,
                        Err(err) => {
                            eprintln!("Can't get mat: {}", err);
                            continue;
                        }
                    };

                    if let Some(t) = nano_track.as_mut() {
                        if let Ok(bbox) = t.update(&mat) {
                            imgproc::rectangle(
                                &mut mat,
                                bbox.unwrap(),
                                Scalar::new(0.0, 255., 0., 0.),
                                2,
                                imgproc::LINE_8,
                                0,
                            ).unwrap();
                        } else {
                            nano_track = None;
                        }
                    }

                    if nano_track.is_none() {
                        let mut input = mat_to_ndarray(&mut mat, 640, 640);
                        let boxes = yolo.infer2(&mut input, w, h);
                        let mut candidate: Option<Rect> = None;

                        if let Some(prev_bbox) = last_bbox {
                            let mut best_iou = 0.0;
                            for b in &boxes {
                                let new_bbox = Rect::new(
                                    b.x1 as i32,
                                    b.y1 as i32,
                                    (b.x2 -b.x1) as i32,
                                    (b.y2 -b.y1) as i32,
                                );
                                let iou_val = iou(&prev_bbox, &new_bbox);
                                if iou_val > best_iou {
                                    best_iou = iou_val;
                                    candidate = Some(new_bbox);
                                }
                            }
                        }

                        if candidate.is_none() {
                            if let Some(first) = boxes.first() {
                                candidate = Some(Rect::new(
                                    first.x1 as i32,
                                    first.y1 as i32,
                                    (first.x2 - first.x1) as i32,
                                    (first.y2 - first.y1) as i32,
                                ));
                            }
                        }

                        if let Some(candidate) = candidate {
                            nano_track = Some(NanoTrack::new(candidate, &mat).unwrap());
                        }
                    }


                    // let _ = imgproc::put_text(
                    //     &mut mat,
                    //     "Rust and OpenCV",
                    //     core::Point::new(30, 50),
                    //     imgproc::FONT_HERSHEY_SIMPLEX,
                    //     1.0,
                    //     core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                    //     2,
                    //     imgproc::LINE_AA,
                    //     false,
                    // );

                    let mut out_buffer = gstreamer::Buffer::with_size((w * h * 3) as usize)
                        .expect("Can't get buffer");
                    {
                        let out_buffer_mut = out_buffer.get_mut().expect("Can't get buffer");
                        let mut out_map = out_buffer_mut.map_writable().expect("Can't get buffer");
                        out_map.copy_from_slice(mat.data_bytes().unwrap());
                    }

                    match appsrc_thread.push_buffer(out_buffer) {
                        Ok(_) => {}
                        Err(err) => {
                            eprintln!("Can't push buffer: {}", err);
                            continue;
                        }
                    }
                }
            }
        }
    });

    let main_loop = gstreamer::glib::MainLoop::new(None, false);
    main_loop.run();

    Ok(())
}
