use aksr::Builder;

use crate::{Hbb, Image, Keypoint, Mask, Obb, Polygon, Prob, Text};
use serde::{Deserialize, Serialize};

/// Container for inference results for each image.
///
/// This struct holds various possible outputs from an image inference process,
/// including probabilities, bounding boxes, keypoints, minimum bounding rectangles,
/// polygons, masks, text annotations, and embeddings.
///
#[derive(Builder, Clone, Default, Serialize, Deserialize)]
pub struct Y {
    texts: Option<Vec<Text>>,
    probs: Option<Vec<Prob>>,
    keypoints: Option<Vec<Keypoint>>,
    keypointss: Option<Vec<Vec<Keypoint>>>,
    hbbs: Option<Vec<Hbb>>,
    obbs: Option<Vec<Obb>>,
    // TODO: Figure out how to serialize polygons
    #[serde(skip)]
    polygons: Option<Vec<Polygon>>,
    // TODO: Figure out how to serialize masks
    #[serde(skip)]
    masks: Option<Vec<Mask>>,
    // TODO: Figure out how to serialize images
    #[serde(skip)]
    images: Option<Vec<Image>>,
}

impl std::fmt::Debug for Y {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Y");
        if let Some(xs) = &self.texts {
            if !xs.is_empty() {
                f.field("Texts", &xs);
            }
        }
        if let Some(xs) = &self.probs {
            f.field("Probs", &xs);
        }
        if let Some(xs) = &self.hbbs {
            if !xs.is_empty() {
                f.field("Hbbs", &xs);
            }
        }
        if let Some(xs) = &self.obbs {
            if !xs.is_empty() {
                f.field("Obbs", &xs);
            }
        }
        if let Some(xs) = &self.keypoints {
            if !xs.is_empty() {
                f.field("Kpts", &xs);
            }
        }
        if let Some(xs) = &self.keypointss {
            if !xs.is_empty() {
                f.field("Kptss", &xs);
            }
        }
        if let Some(xs) = &self.polygons {
            if !xs.is_empty() {
                f.field("Polys", &xs);
            }
        }
        if let Some(xs) = &self.masks {
            if !xs.is_empty() {
                f.field("Masks", &xs);
            }
        }
        if let Some(xs) = &self.images {
            if !xs.is_empty() {
                f.field("Images", &xs);
            }
        }
        f.finish()
    }
}
