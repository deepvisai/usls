//! Implementation of the GLASS model: preprocessing, inference, postprocessing.
use crate::{elapsed_module, Config, Engine, Heatmap, Image, Processor, Xs, Y};
use anyhow::Result;
use image::{GrayImage, Luma};
use log::debug;
use ndarray::Axis;

/// GLASS anomaly detection model struct.
#[derive(Debug)]
pub struct Dinomaly {
    engine: Engine,
    processor: Processor,
}

impl Dinomaly {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;

        let (height, width) = (
            engine.try_height().unwrap_or(&384.into()).opt(),
            engine.try_width().unwrap_or(&384.into()).opt(),
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
        let result = self.engine.run(xs)?;
        debug!("Inference output length: {}", result.len());
        if result.is_empty() {
            debug!("WARNING: Inference returned empty results!");
        } else {
            for (i, tensor) in result.iter().enumerate() {
                debug!("Output tensor {}: shape {:?}", i, tensor.1.shape());
            }
        }
        Ok(result)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("DINOMALY", "visual-preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("DINOMALY", "visual-inference", self.inference(ys)?);
        let ys = elapsed_module!("DINOMALY", "visual-postprocess", self.postprocess(ys)?);
        Ok(ys)
    }

    fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let mut results = Vec::new();

        // Check that we have all expected outputs
        if xs.len() < 4 {
            return Err(anyhow::anyhow!("Expected 4 outputs, got {}", xs.len()));
        }

        let pred_score_tensor = &xs[0]; // Global anomaly score
        let anomaly_map_tensor = &xs[2]; // 🎯 This is the spatial heatmap!

        // For each item in the batch, build a Heatmap from the anomaly map
        for (i, batch_out) in anomaly_map_tensor.axis_iter(Axis(0)).enumerate() {
            // Squeeze optional channel dim: [1, H, W] -> [H, W]
            let raw_map = if batch_out.ndim() == 3 {
                batch_out.index_axis(Axis(0), 0).to_owned()
            } else {
                batch_out.to_owned()
            };

            // Clamp to [0,1] and convert to 8-bit grayscale image
            let raw_map = raw_map.mapv(|v| v.clamp(0.0, 1.0));
            let (h, w) = (raw_map.shape()[0], raw_map.shape()[1]);

            let mut gray = GrayImage::new(w as u32, h as u32);
            for (y, row) in raw_map.outer_iter().enumerate() {
                for (x, &v) in row.iter().enumerate() {
                    gray.put_pixel(x as u32, y as u32, Luma([(v * 255.0) as u8]));
                }
            }

            // Pull the global score for this item and clamp to [0,1]
            let global_score = if pred_score_tensor.ndim() == 1 {
                pred_score_tensor[[i]]
            } else {
                pred_score_tensor[[i, 0]]
            }
            .clamp(0.0, 1.0);

            // ✅ Only store the anomaly map; set confidence = pred score
            let heatmap = Heatmap::from(gray).with_confidence(global_score);
            results.push(Y::default().with_heatmaps(&[heatmap]));

            debug!("Processed item {} with confidence={:.4}", i, global_score);
        }

        Ok(results)
    }
}
