use std::path::Path;
use ndarray::{s, Axis};
use crate::utils::BBox;
use ort::session::Session;
use ort::value::TensorRef;

pub struct YoloV8 {
    session: Session,
}

impl YoloV8 {
    pub fn new() -> ort::Result<Self> {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("yolov8n.onnx");
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }
    
    pub fn infer2(
        &mut self,
        input: &ndarray::Array<f32, ndarray::Dim<[usize; 4]>>,
        img_width: i32,
        img_height: i32,
    ) -> Vec<BBox> {
        //let mut sw = Stopwatch::start_new();
        let outputs = self
            .session
            .run(ort::inputs!["images" => TensorRef::from_array_view(input).unwrap()])
            .unwrap();
        //sw.stop();
        //println!("stop run: {:?}", sw.elapsed().as_millis());

        //sw.restart();
        let output = outputs["output0"]
            .try_extract_array::<f32>()
            .unwrap()
            .t()
            .into_owned();
        //sw.stop();
        //println!("stop try_extract_array: {:?}", sw.elapsed().as_millis());


        let mut boxes = Vec::<BBox>::new();

        //sw.restart();
        let output = output.slice(s![..,..,0]);
        //sw.stop();
        //println!("stop slice: {:?}", sw.elapsed().as_millis());

        //sw.restart();

        for row in output.axis_iter(Axis(0)) {
            // первые 4 значения — bbox
            let xc = row[0usize] / 640.0 * (img_width as f32);
            let yc = row[1usize] / 640.0 * (img_height as f32);
            let w  = row[2usize] / 640.0 * (img_width as f32);
            let h  = row[3usize] / 640.0 * (img_height as f32);

            // ищем максимум среди классов (начиная с индекса 4)
            let mut best_class = 0;
            let mut best_prob = f32::MIN;
            for (i, &val) in row.iter().enumerate().skip(4) {
                if val > best_prob {
                    best_prob = val;
                    best_class = i - 4; // смещение на 4
                }
            }

            if best_prob < 0.5 {
                continue;
            }

            // if best_class == 4 || best_class == 8 || best_class == 2 || best_class == 5 || best_class == 7 {
                boxes.push(BBox {
                    x1: xc - w / 2.0,
                    y1: yc - h / 2.0,
                    x2: xc + w / 2.0,
                    y2: yc + h / 2.0,
                    class_id: best_class,
                    confidence: best_prob,
                });
            // }
        }

        //sw.stop();
        //println!("stop foreach: {:?}", sw.elapsed().as_millis());
        //println!("================================================");
        boxes
    }
}
