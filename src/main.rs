use gstreamer::Pipeline;
use gstreamer::prelude::*;
use opencv::prelude::*;
use opencv::{core, imgproc};

fn main() {
    gstreamer::init().unwrap();

    let pipeline_in_str = "libcamerasrc !\
     video/x-raw,format=BGR,width=640,height=512,framerate=25/1 !appsink name=sink";

    let pipeline_in = gstreamer::parse::launch(pipeline_in_str).expect("Can't launch pipeline");
    let pipeline_in = pipeline_in
        .dynamic_cast::<Pipeline>()
        .expect("Couldn't cast pipeline to pipeline");

    let appsink = pipeline_in
        .by_name("sink")
        .unwrap()
        .dynamic_cast::<gstreamer_app::AppSink>()
        .unwrap();
    appsink.set_property("emit-signals", &true);

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

    appsrc.set_caps(Some(
        &gstreamer::Caps::builder("video/x-raw")
            .field("format", &"BGR")
            .field("width", &640i32)
            .field("height", &512i32)
            .field("framerate", &gstreamer::Fraction::new(25, 1))
            .build(),
    ));

    std::thread::spawn(move || {
        loop {
            match appsink.pull_sample() {
                Ok(sample) => {
                    let buffer = sample.buffer().unwrap();
                    let caps = sample.caps().unwrap();
                    let s = caps.structure(0).unwrap();
                    let width = s.get::<i32>("width").unwrap();
                    let height = s.get::<i32>("height").unwrap();

                    let map = buffer.map_readable().unwrap();
                    let mut data = map.as_slice().to_vec();

                    let mut mat =
                        Mat::new_rows_cols_with_bytes_mut::<u8>(height, width * 3, &mut data)
                            .unwrap();

                    imgproc::put_text(
                        &mut mat,
                        "Rust and OpenCV",
                        core::Point::new(30, 50),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        1.0,
                        core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                        2,
                        imgproc::LINE_AA,
                        false,
                    )
                    .unwrap();

                    let mut out_buffer =
                        gstreamer::Buffer::with_size((width * height * 3) as usize).unwrap();
                    {
                        let out_buffer_mut = out_buffer.get_mut().unwrap();
                        let mut out_map = out_buffer_mut.map_writable().unwrap();
                        out_map.copy_from_slice(mat.data_bytes().unwrap());
                    }

                    let _ = appsrc.push_buffer(out_buffer);
                }
                Err(_) => {
                    break;
                }
            }
        }
    });

    let main_loop = gstreamer::glib::MainLoop::new(None, false);
    main_loop.run();
}
