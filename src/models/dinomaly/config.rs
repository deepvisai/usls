impl crate::Config {
    pub fn dinomaly() -> Self {
        Self::default()
            .with_model_ixx(0, 0, 2.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 392.into())
            .with_model_ixx(0, 3, 392.into())
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_resize_filter("Lanczos3")
            .with_normalize(true)
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_image_mean(&[0.485, 0.456, 0.406])
    }
}
