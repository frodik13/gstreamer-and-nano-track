use opencv::core::{Ptr, Rect, ToInputArray};
use opencv::prelude::*;
use opencv::video::{TrackerVit, TrackerVit_Params};
use std::path::Path;

pub struct VitTracker {
    tracker: Ptr<TrackerVit>,
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

        Ok(VitTracker { tracker })
    }

    pub fn update(&mut self, frame: &Mat) -> opencv::Result<Option<Rect>> {
        let mut bbox = Rect::default();
        let ok = self.tracker.update(&frame, &mut bbox)?;

        let score = self.tracker.get_tracking_score()?;
        println!("Score vit tracker: {:?}", score);

        if score < 0.7 {
            Ok(Some(bbox))
        } else {
            Ok(None)
        }
    }
}
