//! Implementation of the GLASS model: preprocessing, inference, postprocessing.
use crate::{elapsed_module, Config, Engine, Heatmap, Image, Processor, Xs, Y};
use anyhow::Result;
use image::{GrayImage};
use log::debug;
use ndarray::Axis;

/// GLASS anomaly detection model struct.
#[derive(Debug)]
pub struct UniNet {
    engine: Engine,
    processor: Processor,
}

impl UniNet {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;

        let (height, width) = (
            engine.try_height().unwrap_or(&392.into()).opt(),
            engine.try_width().unwrap_or(&392.into()).opt(),
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
        let ys = elapsed_module!("UNINET", "visual-preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("UNINET", "visual-inference", self.inference(ys)?);
        let ys = elapsed_module!("UNINET", "visual-postprocess", self.postprocess(ys)?);
        Ok(ys)
    }

    fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let mut results = Vec::new();

        let pred_score_tensor = &xs[0]; // Global anomaly score
        let anomaly_map_tensor = &xs[2]; // Spatial heatmap

        for (i, batch_out) in anomaly_map_tensor.axis_iter(Axis(0)).enumerate() {
            // batch_out is now [1, 392, 392] for each batch item
            // Skip the channel dimension and get to [392, 392]
            let map_2d = batch_out.index_axis(Axis(0), 0);

            let (height, width) = (map_2d.shape()[0], map_2d.shape()[1]);

            // Flatten and convert to pixels
            let pixels: Vec<u8> = map_2d
                .iter() // or use .flatten() if you prefer
                .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                .collect();

            let gray = GrayImage::from_raw(width as u32, height as u32, pixels)
                .ok_or_else(|| anyhow::anyhow!("Failed to create image"))?;

            // Get score for this batch item
            let global_score = if pred_score_tensor.ndim() == 1 {
                pred_score_tensor[[i]]
            } else {
                pred_score_tensor[[i, 0]]
            }
            .clamp(0.0, 1.0);

            let heatmap = Heatmap::from(gray).with_confidence(global_score);
            results.push(Y::default().with_heatmaps(&[heatmap]));
        }

        Ok(results)
    }
}
