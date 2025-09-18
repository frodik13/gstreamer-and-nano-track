use ndarray::{Array, Array4};
use opencv::Result;
use opencv::core::{Rect, ToInputArray, ToInputOutputArray};
use opencv::prelude::*;
use opencv::{core, imgproc};
use std::fs;

#[derive(Debug)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub confidence: f32,
}

pub fn mat_to_ndarray(
    frame: &impl ToInputArray,
    width: i32,
    height: i32,
) -> Array<f32, ndarray::Dim<[usize; 4]>> {
    // 1) resize -> 640x640
    let mut resized = Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        core::Size { width, height },
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
    .unwrap();

    // 2) BGR -> RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(
        &resized,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .unwrap();

    // 3) uint8 -> float32 и нормализация /255
    let mut rgb_float = Mat::default();
    rgb.convert_to(&mut rgb_float, core::CV_32F, 1.0 / 255.0, 0.0)
        .unwrap();

    // 4) берем буфер как &[Vec3f] — правильно для CV_32FC3
    //    (data_typed::<f32>() не годится для 3-канального Mat)
    let vec3s = rgb_float
        .data_typed::<core::Vec3f>()
        .expect("data_typed::<Vec3f>() failed — unexpected Mat type");

    let rows = rgb_float.rows() as usize;
    let cols = rgb_float.cols() as usize;
    let num_pixels = rows * cols;

    // 5) создаём выходной буфер в формате NCHW сразу
    let mut out = vec![0f32; 1 * 3 * num_pixels];

    // Заполняем: для пикселя i (row-major H*W) тройка vec3s[i] = [R,G,B]
    // нужный индекс в NCHW: channel * (H*W) + i
    for (i, v) in vec3s.iter().enumerate() {
        out[0 * num_pixels + i] = v[0]; // R
        out[1 * num_pixels + i] = v[1]; // G
        out[2 * num_pixels + i] = v[2]; // B
    }

    Array4::from_shape_vec((1, 3, rows, cols), out).unwrap()
}

pub fn center_crop(frame: &impl ToInputArray, crop_size: i32) -> Result<Mat> {
    let input_array = frame.input_array().expect("frame.input_array() failed");
    let mat = input_array
        .get_mat(0)
        .expect("input_array().get_mat(0) failed");

    let rows = mat.rows();
    let cols = mat.cols();

    let x = (cols - crop_size) / 2;
    let y = (rows - crop_size) / 2;

    let roi = core::Rect::new(x, y, crop_size, crop_size);
    let cropped = Mat::roi(&mat, roi).expect("roi() failed");

    Ok(cropped.clone_pointee())
}

/*pub fn expand_roi(frame: &impl ToInputArray, prev_roi: Rect, expand: i32) -> Result<Mat> {
    let input_array = frame.input_array()?;
    let mat = input_array.get_mat(0)?;

    let rows = mat.rows();
    let cols = mat.cols();

    // Новый ROI с расширением
    let mut x = prev_roi.x - expand;
    let mut y = prev_roi.y - expand;
    let mut w = prev_roi.width + expand * 2;
    let mut h = prev_roi.height + expand * 2;

    // Корректируем границы по размеру изображения
    if x < 0 {
        w += x; // уменьшаем ширину
        x = 0;
    }
    if y < 0 {
        h += y; // уменьшаем высоту
        y = 0;
    }
    if x + w > cols {
        w = cols - x;
    }
    if y + h > rows {
        h = rows - y;
    }

    // Если размеры ушли в ноль или отрицательные → пустой ROI
    if w <= 0 || h <= 0 {
        return Ok(mat);
        // return Err(opencv::Error::new(
        //     opencv::core::StsBadArg,
        //     "Expanded ROI is invalid".to_string(),
        // ));
    }

    let roi = Rect::new(x, y, w, h);
    let cropped = Mat::roi(&mat, roi)?;

    Ok(cropped.clone_pointee())
}*/

pub fn draw_bboxes(frame: &mut Mat, bboxes: &[BBox], labels: &[&str]) -> opencv::Result<()> {
    for bbox in bboxes {
        let rect = core::Rect {
            x: bbox.x1 as i32,
            y: bbox.y1 as i32,
            width: (bbox.x2 - bbox.x1) as i32,
            height: (bbox.y2 - bbox.y1) as i32,
        };

        imgproc::rectangle(
            frame,
            rect,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            opencv::imgproc::LINE_8,
            0,
        )?;

        let label = format!("{} -> {:.2}", labels[bbox.class_id], bbox.confidence);
        imgproc::put_text(
            frame,
            &label,
            core::Point::new(rect.x, rect.y - 5),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }

    Ok(())
}

pub fn iou(a: &Rect, b: &Rect) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);

    let inter_area = ((x2 - x1).max(0) * (y2 - y1).max(0)) as f32;
    let a_area = (a.width * a.height) as f32;
    let b_area = (b.width * b.height) as f32;

    inter_area / (a_area + b_area - inter_area)
}

pub fn get_cpu_usage() -> f32 {
    let loadavg = fs::read_to_string("/proc/loadavg").expect("Can't read /proc/loadavg");
    let parts: Vec<&str> = loadavg.trim().split(' ').collect();
    if let Some(cpu_usage) = parts.get(0) {
        return cpu_usage.parse::<f32>().unwrap_or(0.0) * 100.0 / 4.0;
    }

    0.0
}

pub fn get_mem_usage() -> f32 {
    let mem_info = fs::read_to_string("/proc/meminfo").expect("Can't read /proc/meminfo");
    let mut total: f32 = 0.0;
    let mut free: f32 = 0.0;

    for line in mem_info.lines() {
        if line.starts_with("MemTotal:") {
            total = line
                .split_whitespace()
                .nth(1)
                .unwrap_or("0")
                .parse::<f32>()
                .unwrap_or(0.0);
        }
        if line.starts_with("MemAvailable:") {
            free = line
                .split_whitespace()
                .nth(1)
                .unwrap_or("0")
                .parse::<f32>()
                .unwrap_or(0.0);
        }
    }

    if total > 0.0 {
        return (1.0 - free / total) * 100.0;
    }

    0.0
}

pub fn get_cpu_temp() -> f32 {
    let temp_str = fs::read_to_string("/sys/class/thermal/thermal_zone0/temp").unwrap_or_default();
    let temp_milli: f32 = temp_str.trim().parse::<f32>().unwrap_or(0.0);
    temp_milli / 1000.0
}

pub fn expand_roi_rect(frame: &impl ToInputArray, prev_roi: Rect, expand: i32) -> Result<Rect> {
    let input_array = frame.input_array()?;
    let mat = input_array.get_mat(0)?;

    let rows = mat.rows();
    let cols = mat.cols();

    let mut x = prev_roi.x - expand;
    let mut y = prev_roi.y - expand;
    let mut w = prev_roi.width + expand * 2;
    let mut h = prev_roi.height + expand * 2;

    if x < 0 {
        w += x;
        x = 0;
    }
    if y < 0 {
        h += y;
        y = 0;
    }
    if x + w > cols {
        w = cols - x;
    }
    if y + h > rows {
        h = rows - y;
    }

    if w <= 0 || h <= 0 {
        return Err(opencv::Error::new(
            opencv::core::StsBadArg,
            "Expanded ROI is invalid".to_string(),
        ));
    }

    Ok(Rect::new(x, y, w, h))
}

pub fn expand_roi(frame: &impl ToInputArray, prev_roi: Rect, expand: i32) -> Result<(Mat, Rect)> {
    let roi = expand_roi_rect(frame, prev_roi, expand)?;
    let input_array = frame.input_array()?;
    let mat = input_array.get_mat(0)?;
    let cropped = Mat::roi(&mat, roi)?;
    Ok((cropped.clone_pointee(), roi))
}