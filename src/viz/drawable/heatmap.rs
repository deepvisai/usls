use crate::{Drawable, HeatMap};
use anyhow::Result;
use image::Rgba;
use image::RgbaImage;

use crate::{DrawContext, Style};

impl Drawable for Vec<HeatMap> {
    fn get_global_style<'a>(&self, ctx: &'a DrawContext) -> Option<&'a Style> {
        ctx.heatmap_style
    }

    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        // For each heatmap in the vec, just call its draw()
        for heatmap in self {
            heatmap.draw(ctx, canvas)?;
        }
        Ok(())
    }
}

impl Drawable for HeatMap {
    fn get_local_style(&self) -> Option<&Style> {
        self.style()
    }

    fn get_global_style<'a>(&self, ctx: &'a DrawContext) -> Option<&'a Style> {
        ctx.mask_style // Reuse for now
    }

    fn get_id(&self) -> Option<usize> {
        self.id()
    }

    fn draw(&self, ctx: &DrawContext, canvas: &mut RgbaImage) -> Result<()> {
        let style = ctx.update_style(
            self.get_local_style(),
            self.get_global_style(ctx),
            self.get_id(),
        );

        let alpha = style.color_fill_alpha().unwrap_or(120);
        let (w, h) = canvas.dimensions();
        let (mw, mh) = self.map().dimensions();

        let x_offset = ((w as i32 - mw as i32) / 2).max(0) as u32;
        let y_offset = ((h as i32 - mh as i32) / 2).max(0) as u32;

        // Overlay buffer
        let mut overlay = RgbaImage::new(w, h);

        let colormap_opt = style.colormap256();

        for y in 0..mh {
            for x in 0..mw {
                let value = self.map().get_pixel(x, y)[0]; // u8 in 0–255
                let color = if let Some(colormap) = colormap_opt {
                    let rgb = colormap.data()[value as usize].rgb(); // (u8, u8, u8)
                    Rgba([rgb.0, rgb.1, rgb.2, alpha])
                } else {
                    // Default: green (low) → yellow → red (high)
                    let norm = value as f32 / 255.0;
                    let r = if norm < 0.5 { 2.0 * norm } else { 1.0 };
                    let g = if norm < 0.5 { 1.0 } else { 2.0 * (1.0 - norm) };
                    let b = 0.0;
                    Rgba([
                        (r * 255.0) as u8,
                        (g * 255.0) as u8,
                        (b * 255.0) as u8,
                        alpha,
                    ])
                };

                overlay.put_pixel(x + x_offset, y + y_offset, color);
            }
        }

        image::imageops::overlay(canvas, &overlay, 0, 0);
        Ok(())
    }
}
