use opencv::core::{Ptr, Rect, ToInputArray};
use opencv::prelude::*;
use opencv::video::{TrackerVit, TrackerVit_Params};
use std::path::Path;
use ticky::Stopwatch;

pub struct VitTracker {
    tracker: Ptr<TrackerVit>,
    second_tracker: Ptr<TrackerVit>,
    last_bbox: Option<Rect>,
    last_frame: Option<Mat>,
}

impl VitTracker {
    pub fn new(initial_bbox: Rect, frame: &impl ToInputArray) -> opencv::Result<Self> {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("object_tracking_vittrack_2023sep_int8bq.onnx");
        let mut param = TrackerVit_Params::default()?;
        param.set_backend(opencv::dnn::DNN_BACKEND_OPENCV);
        param.set_target(opencv::dnn::DNN_TARGET_CPU);
        param.set_net(model_path.to_str().unwrap());

        let mut tracker = TrackerVit::create(&param)?;
        tracker.init(frame, initial_bbox)?;

        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("object_tracking_vittrack_2023sep.onnx");

        let mut param = TrackerVit_Params::default()?;
        param.set_backend(opencv::dnn::DNN_BACKEND_OPENCV);
        param.set_target(opencv::dnn::DNN_TARGET_CPU);
        param.set_net(model_path.to_str().unwrap());

        let second_tracker = TrackerVit::create(&param)?;

        Ok(VitTracker {
            tracker,
            second_tracker,
            last_bbox: Some(initial_bbox),
            last_frame: Some(frame.input_array()?.get_mat(0)?),
        })
    }

    pub fn update(&mut self, frame: &Mat) -> opencv::Result<Option<Rect>> {
        let mut sw = Stopwatch::start_new();
        let mut bbox = Rect::default();
        let ok = self.tracker.update(&frame, &mut bbox)?;

        let score = self.tracker.get_tracking_score()?;
        sw.stop();
        println!(
            "Score vit tracker: {:?}. Inference time: {} ms",
            score,
            sw.elapsed.as_millis()
        );

        if score >= 0.45 {
            self.last_bbox = Some(bbox);
            self.last_frame = Some(frame.clone());
            Ok(Some(bbox))
        } else {
            println!("initial second tracker");

            if let Some(last_bbox) = self.last_bbox {
                if let Some(last_frame) = &self.last_frame {
                    sw.restart();
                    self.second_tracker.init(last_frame, last_bbox)?;
                    sw.stop();
                    println!("init second tracker time: {} ms", sw.elapsed.as_millis());

                    sw.restart();
                    self.second_tracker.update(frame, &mut bbox)?;
                    let score = self.second_tracker.get_tracking_score()?;
                    sw.stop();
                    println!(
                        "\tScore second tracker: {:?}. Inference time: {} ms",
                        score,
                        sw.elapsed.as_millis()
                    );

                    if score >= 0.55 {
                        self.last_bbox = Some(bbox);
                        self.last_frame = Some(frame.clone());
                        Ok(Some(bbox))
                    } else {
                        self.return_none()
                    }
                } else {
                    self.return_none()
                }
            } else {
                self.return_none()
            }
        }
    }

    fn return_none(&mut self) -> opencv::Result<Option<Rect>> {
        self.last_bbox = None;
        self.last_frame = None;
        Ok(None)
    }
}
