//! Implementation of the GLASS model: preprocessing, inference, postprocessing.
use crate::{elapsed, Engine, Image, Mask, MinOptMax, Options, Prob, Processor, Ts, Xs, Y};
use anyhow::Result;
use image::{imageops::FilterType, GrayImage, Luma};
use ndarray::Axis;

/// GLASS anomaly detection model struct.
#[derive(Debug)]
pub struct GLASS {
    engine: Engine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    ts: Ts,
    processor: Processor,
    spec: String,
}

impl TryFrom<Options> for GLASS {
    type Error = anyhow::Error;

    fn try_from(options: Options) -> Result<Self, Self::Error> {
        Self::new(options)
    }
}

impl GLASS {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;

        let (batch, height, width, ts, spec) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&256.into()).opt(),
            engine.try_width().unwrap_or(&256.into()).opt(),
            engine.ts.clone(),
            engine.spec().to_owned(),
        );

        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height: height.into(),
            width: width.into(),
            batch: batch.into(),
            processor,
            ts,
            spec,
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
        let ys = elapsed!("preprocess", self.ts, { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });

        Ok(ys)
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }

    fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let mut results = Vec::new();
        let output_tensor = &xs[0]; // shape: [B, 1, 256, 256]

        for batch_out in output_tensor.axis_iter(Axis(0)) {
            let raw_map = batch_out
                .index_axis(Axis(0), 0)
                .to_owned()
                .mapv(|v| v.clamp(0.0, 1.0));
            let max_score = raw_map.iter().copied().fold(0.0, f32::max);

            let (h, w) = (raw_map.shape()[0], raw_map.shape()[1]);
            let mut small = GrayImage::new(w as u32, h as u32);
            for (y, row) in raw_map.outer_iter().enumerate() {
                for (x, &v) in row.iter().enumerate() {
                    small.put_pixel(x as u32, y as u32, Luma([(v * 255.0) as u8]));
                }
            }
            let heatmap = image::imageops::resize(&small, w as u32, h as u32, FilterType::Triangle);

            let mask = Mask::default()
                .with_mask(heatmap)
                .with_name("anomaly")
                .with_confidence(max_score);
            let prob = Prob::default()
                .with_name("anomaly_score")
                .with_confidence(max_score);
            results.push(Y::default().with_masks(&[mask]).with_probs(&[prob]));
        }
        Ok(results)
    }
}
