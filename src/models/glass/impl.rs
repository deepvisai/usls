//! Implementation of the GLASS model: preprocessing, inference, postprocessing.
use crate::{elapsed_module, Config, Engine, Image, Processor, Xs, Y, Heatmap};
use anyhow::Result;
use image::{GrayImage, Luma};
use log::debug;
use ndarray::Axis;

/// GLASS anomaly detection model struct.
#[derive(Debug)]
pub struct GLASS {
    engine: Engine,
    processor: Processor,
}

impl GLASS {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;

        let (height, width) = (
            engine.try_height().unwrap_or(&256.into()).opt(),
            engine.try_width().unwrap_or(&256.into()).opt(),
        );

        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self { engine, processor })
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

        for (i, batch_out) in output_tensor.axis_iter(Axis(0)).enumerate() {

            let raw_map = batch_out.to_owned().mapv(|v| v.clamp(0.0, 1.0));
            let max_score = raw_map.iter().copied().fold(0.0, f32::max);
            let (h, w) = (raw_map.shape()[0], raw_map.shape()[1]);

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

            let heatmap = Heatmap::from(small).with_confidence(max_score);

            results.push(
                Y::default()
                    .with_heatmaps(&[heatmap])
            );

            debug!("Finished processing batch {}", i);
        }

        Ok(results)
    }
}
