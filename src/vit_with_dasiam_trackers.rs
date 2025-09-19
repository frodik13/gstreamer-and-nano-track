use opencv::core::{Mat, Ptr, Rect, ToInputArray};
use opencv::hub_prelude::TrackerTrait;
use opencv::prelude::*;
use opencv::video::{TrackerDaSiamRPN, TrackerDaSiamRPN_Params, TrackerVit, TrackerVit_Params};
use std::path::Path;

pub struct VitWithDaSiamTracker {
    first_tracker: Ptr<TrackerVit>,
    second_tracker: Ptr<TrackerDaSiamRPN>,
    last_bbox: Option<Rect>,
    last_frame: Option<Mat>,
}

impl VitWithDaSiamTracker {
    pub fn new(initial_bbox: Rect, frame: &impl ToInputArray) -> opencv::Result<Self> {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("object_tracking_vittrack_2023sep.onnx");
        let mut param = TrackerVit_Params::default()?;
        param.set_backend(opencv::dnn::DNN_BACKEND_OPENCV);
        param.set_target(opencv::dnn::DNN_TARGET_CPU);
        param.set_net(model_path.to_str().unwrap());

        let mut first_tracker = TrackerVit::create(&param)?;
        first_tracker.init(frame, initial_bbox)?;

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
        param.set_backend(opencv::dnn::DNN_BACKEND_OPENCV);
        param.set_target(opencv::dnn::DNN_TARGET_CPU);
        let second_tracker = TrackerDaSiamRPN::create(&param)?;

        Ok(VitWithDaSiamTracker {
            first_tracker,
            second_tracker,
            last_bbox: Some(initial_bbox.clone()),
            last_frame: Some(frame.clone().input_array()?.get_mat(0)?),
        })
    }

    pub fn update(&mut self, frame: &Mat) -> opencv::Result<Option<Rect>> {
        let mut bbox = Rect::default();
        let ok = self.first_tracker.update(&frame, &mut bbox)?;

        let score = self.first_tracker.get_tracking_score()?;
        println!("Score vit tracker: {:?}", score);

        if score >= 0.45 {
            self.last_frame = Some(frame.clone());
            self.last_bbox = Some(bbox.clone());
            Ok(Some(bbox))
        } else {
            println!("Init second tracker");
            if let Some(last_bbox) = self.last_bbox {
                if let Some(last_frame) = &self.last_frame {
                    self.second_tracker.init(last_frame, last_bbox)?;
                    self.second_tracker.update(frame, &mut bbox)?;

                    let score = self.second_tracker.get_tracking_score()?;
                    println!("\tScore DaSiam tracker: {:?}", score);
                    if score >= 0.5 {
                        self.last_frame = Some(frame.clone());
                        self.last_bbox = Some(bbox.clone());
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
