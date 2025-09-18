use opencv::core::{Ptr, Rect};
use opencv::tracking::{TrackerKCF, TrackerKCF_Params};
use opencv::prelude::*;

pub struct KcfTracker {
    tracker: Ptr<TrackerKCF>
}

impl KcfTracker {
    pub fn new(initial_bbox: Rect, frame: &Mat) -> opencv::Result<Self> {
        let params = TrackerKCF_Params {
            detect_thresh: 0.07,       // 0.5
            sigma: 1.043590774305246,  // 0.2
            lambda: 3e-07,             // 0.0001
            interp_factor: 0.065,      // 0.075
            output_sigma_factor: 0.03, // 0.0625
            pca_learning_rate: 0.5,    // 0.15
            resize: true,              // true
            split_coeff: false,        // true
            wrap_kernel: false,        // false
            compress_feature: false,   // true
            max_patch_size: 3200,      // 6400
            compressed_size: 1,        // 2
            desc_pca: 2,               // 2
            desc_npca: 1,              // 1
        };
        // let def_param = TrackerKCF_Params::default()?;
        let mut tracker = TrackerKCF::create(params)?;
        tracker.init(frame, initial_bbox)?;
        Ok(Self { tracker })
    }

    pub fn update(&mut self, frame: &Mat) -> opencv::Result<Option<Rect>> {
        let mut bbox = Rect::default();
        let ok = self.tracker.update(&frame, &mut bbox)?;
        if ok { Ok(Some(bbox)) } else { Ok(None) }
    }
}