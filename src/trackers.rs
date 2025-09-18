use std::path::Path;
use opencv::core::{Ptr, Rect, ToInputArray};
use opencv::prelude::*;
use opencv::video::{TrackerDaSiamRPN, TrackerDaSiamRPN_Params, TrackerNano, TrackerNano_Params, TrackerNano_ParamsTrait, TrackerTrait};

pub struct NanoTrack {
    tracker: Ptr<TrackerNano>,
    second_tracker: Ptr<TrackerDaSiamRPN>,
    last_bbox: Option<Rect>,
}

impl NanoTrack {
    pub fn new(initial_bbox: Rect, frame: &impl ToInputArray) -> opencv::Result<Self>
    where
        Self: Sized,
    {
        let head = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("nanotrack_head_sim.onnx");
        let backbone = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("nanotrack_backbone_sim.onnx");

        let mut param = TrackerNano_Params::default()?;
        param.set_backbone(backbone.to_str().unwrap());
        param.set_neckhead(head.to_str().unwrap());

        let mut tracker = TrackerNano::create(&param)?;
        tracker.init(frame, initial_bbox)?;

        let model_siam_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("dasiamrpn_model.onnx");

        let cls1 = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("dasiamrpn_kernel_cls1.onnx");

        let r1 = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("dasiamrpn_kernel_r1.onnx");

        let mut param = TrackerDaSiamRPN_Params::default()?;
        param.set_model(model_siam_path.to_str().unwrap());
        param.set_kernel_cls1(cls1.to_str().unwrap());
        param.set_kernel_r1(r1.to_str().unwrap());
        let second_tracker = TrackerDaSiamRPN::create(&param)?;

        Ok(Self { tracker, second_tracker, last_bbox: Some(initial_bbox) })
    }

    pub fn update(&mut self, frame: &impl ToInputArray) -> opencv::Result<Option<Rect>> {
        let mut bbox = Rect::default();
        // let mut sw = Stopwatch::start_new();
        let ok = self.tracker.update(frame, &mut bbox)?;
        //sw.stop();
        // println!("updated {}", sw.elapsed.as_millis());
        let v = self.tracker.get_tracking_score()?;
        println!("get tracking score: {}", v);
        if v < 0.87 {
            println!("init second_tracker");
            let last_bbox = match self.last_bbox {
                None => {bbox}
                Some(b) => {b}
            };
            self.second_tracker.init(frame, last_bbox)?;
            let ok = self.second_tracker.update(frame, &mut bbox)?;

            let v = self.second_tracker.get_tracking_score()?;
            return if v < 0.7 {
                Ok(None)
            } else {
                Ok(Some(bbox))
            }
        }
        if ok {
            self.last_bbox = Some(bbox);
            Ok(Some(bbox))
        } else {
            self.last_bbox = None;
            Ok(None)
        }
    }
}
