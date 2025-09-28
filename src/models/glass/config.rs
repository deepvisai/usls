impl crate::Config {
    pub fn glass() -> Self {
        Self::default()
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 384.into())
            .with_model_ixx(0, 3, 384.into())
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_resize_filter("CatmullRom")
            .with_normalize(true)
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
    }
}
