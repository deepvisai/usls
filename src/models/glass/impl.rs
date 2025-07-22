//! Implementation of the GLASS model: preprocessing, inference, postprocessing.
use crate::{elapsed_module, Config, Engine, Image, Prob, Processor, Xs, Y};
use anyhow::Result;
use image::{imageops::FilterType, GrayImage, Luma};
use log::debug;
use ndarray::Axis;

/// GLASS anomaly detection model struct.
#[derive(Debug)]
pub struct GLASS {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: Processor,
}

impl GLASS {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;

        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&288.into()).opt(),
            engine.try_width().unwrap_or(&288.into()).opt(),
        );

        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            processor,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        let x = self.processor.process_images(xs)?;
        Ok(x.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("GLASS", "visual-preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("GLASS", "visual-inference", self.inference(ys)?);
        let ys = elapsed_module!("GLASS", "visual-postprocess", self.postprocess(ys)?);
        Ok(ys)
    }

    fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let mut results = Vec::new();
        let output_tensor = &xs[0]; // shape: [B, H, W] or [B, 1, H, W]
        println!("Postprocess input shape: {:?}", output_tensor.shape());

        debug!("Output tensor shape: {:?}", output_tensor.shape());

        for (i, batch_out) in output_tensor.axis_iter(Axis(0)).enumerate() {
            debug!("Processing batch index: {}", i);
            debug!("Batch tensor shape: {:?}", batch_out.shape());

            let raw_map = batch_out.to_owned().mapv(|v| v.clamp(0.0, 1.0));
            let max_score = raw_map.iter().copied().fold(0.0, f32::max);

            let sum: f32 = raw_map.iter().copied().sum();
            let count = raw_map.len();
            let mean_score = if count > 0 { sum / count as f32 } else { 0.0 };

            debug!("Max anomaly score: {:.4}", max_score);

            let (h, w) = (raw_map.shape()[0], raw_map.shape()[1]);
            debug!("Heatmap size: {}x{}", w, h);

            let mut small = GrayImage::new(w as u32, h as u32);
            for (y, row) in raw_map.outer_iter().enumerate() {
                for (x, &v) in row.iter().enumerate() {
                    let pixel_value = (v * 255.0) as u8;
                    small.put_pixel(x as u32, y as u32, Luma([pixel_value]));

                    if x == 0 && y < 3 {
                        debug!("Row {}: pixel[0] anomaly = {:.4}", y, v);
                    }
                }
            }

            let heatmap = image::imageops::resize(&small, 900, 900, FilterType::Triangle);

            let peak_prob = Prob::default()
                .with_name("peak_anomaly_score")
                .with_id(0)
                .with_confidence(max_score);

            let mean_prob = Prob::default()
                .with_name("mean_anomaly_score")
                .with_id(1)
                .with_confidence(mean_score);

            results.push(
                Y::default()
                    .with_images(&[heatmap.into()])
                    .with_probs(&[peak_prob, mean_prob]),
            );

            debug!("Finished processing batch {}", i);
        }

        Ok(results)
    }
}
