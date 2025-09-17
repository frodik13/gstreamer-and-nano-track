use std::path::Path;
use opencv::core::{Ptr, Rect, ToInputArray};
use opencv::prelude::*;
use opencv::video::{TrackerNano, TrackerNano_Params, TrackerNano_ParamsTrait, TrackerTrait};

pub struct NanoTrack {
    tracker: Ptr<TrackerNano>,
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
        Ok(Self { tracker })
    }

    pub fn update(&mut self, frame: &impl ToInputArray) -> opencv::Result<Option<Rect>> {
        let mut bbox = Rect::default();
        // let mut sw = Stopwatch::start_new();
        let ok = self.tracker.update(frame, &mut bbox)?;
        //sw.stop();
        // println!("updated {}", sw.elapsed.as_millis());
        let v = self.tracker.get_tracking_score()?;
        println!("get tracking score: {}", v);
        if v < 0.3 {
            return Ok(None);
        }
        if ok { Ok(Some(bbox)) } else { Ok(None) }
    }
}