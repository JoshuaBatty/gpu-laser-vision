//! PyTorch-backed YOLO instance-segmentation inference.
//!
//! This module owns the TorchScript model and the complete neural-vision path:
//! input preparation, inference, person-mask decoding, and contour extraction.

use anyhow::{Context, Result, bail};
use image::{GrayImage, RgbImage, RgbaImage};
use std::path::Path;
use tch::{CModule, Device, IValue, Kind, Tensor};

const MODEL_IMAGE_SIZE: u32 = 640;
const PERSON_CLASS_INDEX: i64 = 0;
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.25;
const CONTOUR_COLOR_SEARCH_RADIUS: i32 = 8;

/// Default TorchScript artifact produced by `scripts/export_yolo.py`.
pub const DEFAULT_MODEL_PATH: &str = "assets/yolo11n-seg.torchscript";

/// Display-ready result from one YOLO segmentation pass.
pub struct YoloFrame {
    /// Binary mask for the most confident detected person.
    pub person_mask: GrayImage,
    /// One-pixel outline extracted from `person_mask`.
    pub contour: GrayImage,
    /// Row-major indices of every non-zero pixel in `contour`.
    pub contour_pixels: Vec<usize>,
    /// Source-frame colours retained only along `contour`.
    pub colored_contour: RgbImage,
    /// Confidence of the selected person detection, if one passed the threshold.
    pub confidence: Option<f32>,
}

/// Loaded YOLO TorchScript model and its inference device.
pub struct YoloSegmenter {
    model: CModule,
    device: Device,
    input: Tensor,
    source_dimensions: Option<(u32, u32)>,
    confidence_threshold: f32,
}

impl YoloSegmenter {
    /// Loads a TorchScript segmentation model onto the first CUDA device.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_file() {
            bail!(
                "YOLO model not found at {}; generate it with `uv run scripts/export_yolo.py`",
                path.display()
            );
        }

        anyhow::ensure!(
            tch::Cuda::is_available(),
            "LibTorch cannot access CUDA; refusing to run YOLO inference on the CPU"
        );
        let device = Device::Cuda(0);
        let mut model = CModule::load_on_device(path, device)
            .with_context(|| format!("loading YOLO model from {}", path.display()))?;
        model.set_eval();
        tch::Cuda::cudnn_set_benchmark(true);
        let input = Tensor::full(
            [1, 3, MODEL_IMAGE_SIZE as i64, MODEL_IMAGE_SIZE as i64],
            114.0 / 255.0,
            (Kind::Float, device),
        );

        println!("YOLO inference device: {device:?}");
        Ok(Self {
            model,
            device,
            input,
            source_dimensions: None,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
        })
    }

    /// Segments the strongest person detection and maps its mask to the source frame.
    pub fn infer(&mut self, rgba: &RgbaImage) -> Result<YoloFrame> {
        let dimensions = rgba.dimensions();
        if self
            .source_dimensions
            .is_some_and(|previous| previous != dimensions)
        {
            let _ = self.input.fill_(114.0 / 255.0);
        }
        self.source_dimensions = Some(dimensions);

        let transform = prepare_input(rgba, self.device, &self.input)?;
        let output = tch::no_grad(|| {
            self.model
                .forward_is(&[IValue::Tensor(self.input.shallow_clone())])
        })
        .context("running YOLO TorchScript inference")?;
        let (predictions, prototypes) = segmentation_outputs(output)?;

        decode_person(
            predictions,
            prototypes,
            transform,
            self.confidence_threshold,
            rgba,
        )
    }
}

#[derive(Clone, Copy)]
struct LetterboxTransform {
    source_width: u32,
    source_height: u32,
    resized_width: u32,
    resized_height: u32,
    scale: f32,
    pad_x: u32,
    pad_y: u32,
}

#[derive(Clone, Copy)]
struct MaskBounds {
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
}

fn prepare_input(rgba: &RgbaImage, device: Device, input: &Tensor) -> Result<LetterboxTransform> {
    let (source_width, source_height) = rgba.dimensions();
    anyhow::ensure!(
        source_width > 0 && source_height > 0,
        "YOLO input image is empty"
    );

    let scale = (MODEL_IMAGE_SIZE as f32 / source_width as f32)
        .min(MODEL_IMAGE_SIZE as f32 / source_height as f32);
    let resized_width = (source_width as f32 * scale).round() as u32;
    let resized_height = (source_height as f32 * scale).round() as u32;
    let pad_x = (MODEL_IMAGE_SIZE - resized_width) / 2;
    let pad_y = (MODEL_IMAGE_SIZE - resized_height) / 2;

    let source = Tensor::f_from_slice(rgba.as_raw())
        .context("creating YOLO input tensor")?
        .view([1, source_height as i64, source_width as i64, 4])
        .to_device(device)
        .narrow(3, 0, 3)
        .permute([0, 3, 1, 2])
        .to_kind(Kind::Float)
        / 255.0;
    let resized = source.upsample_bilinear2d(
        [resized_height as i64, resized_width as i64],
        false,
        None,
        None,
    );
    input
        .narrow(2, pad_y as i64, resized_height as i64)
        .narrow(3, pad_x as i64, resized_width as i64)
        .copy_(&resized);

    Ok(LetterboxTransform {
        source_width,
        source_height,
        resized_width,
        resized_height,
        scale,
        pad_x,
        pad_y,
    })
}

fn segmentation_outputs(output: IValue) -> Result<(Tensor, Tensor)> {
    let IValue::Tuple(mut values) = output else {
        bail!("YOLO segmentation model returned a non-tuple output");
    };
    if values.len() != 2 {
        bail!(
            "YOLO segmentation model returned {} outputs; expected predictions and prototypes",
            values.len()
        );
    }

    let prototypes = tensor_output(values.pop().expect("output length checked"), "prototypes")?;
    let predictions = tensor_output(values.pop().expect("output length checked"), "predictions")?;
    Ok((predictions, prototypes))
}

fn decode_person(
    predictions: Tensor,
    prototypes: Tensor,
    transform: LetterboxTransform,
    confidence_threshold: f32,
    source: &RgbaImage,
) -> Result<YoloFrame> {
    let (batch_size, prediction_channels, candidate_count) = predictions
        .size3()
        .context("reading YOLO prediction dimensions")?;
    let (prototype_batch, mask_channels, mask_height, mask_width) = prototypes
        .size4()
        .context("reading YOLO prototype dimensions")?;
    anyhow::ensure!(
        batch_size == 1 && prototype_batch == 1,
        "YOLO decoder expects a batch size of one"
    );
    anyhow::ensure!(candidate_count > 0, "YOLO produced no detection candidates");

    let class_count = prediction_channels - 4 - mask_channels;
    anyhow::ensure!(
        class_count > PERSON_CLASS_INDEX,
        "YOLO output does not contain the COCO person class"
    );

    let predictions = predictions.get(0);
    let person_scores = predictions.get(4 + PERSON_CLASS_INDEX);
    let candidate_index = person_scores.argmax(0, false).unsqueeze(0);
    let detection = predictions.index_select(1, &candidate_index).squeeze_dim(1);
    let metadata = detection
        .narrow(0, 0, 5)
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous();
    let metadata = Vec::<f32>::try_from(&metadata).context("copying YOLO detection metadata")?;
    let [center_x, center_y, box_width, box_height, confidence] = metadata.as_slice() else {
        bail!("YOLO detection metadata has an unexpected length");
    };
    let confidence = *confidence;
    if !confidence.is_finite() || confidence < confidence_threshold {
        return Ok(empty_frame(transform));
    }

    // The exported segmentation head is [xywh, class scores, mask coefficients].
    // Selecting one strongest person keeps this first live visualization simple;
    // multi-person selection and NMS can be added when the product needs them.
    let coefficients = detection.narrow(0, 4 + class_count, mask_channels);
    let prototype_logits = prototypes
        .get(0)
        .view([mask_channels, mask_height * mask_width]);
    let mask_logits =
        coefficients
            .unsqueeze(0)
            .matmul(&prototype_logits)
            .view([1, 1, mask_height, mask_width]);

    let (person_mask, mask_bounds) = restore_mask(
        mask_logits,
        [*center_x, *center_y, *box_width, *box_height],
        transform,
    )?;
    let (contour, contour_pixels) = extract_contour(&person_mask, mask_bounds);
    let colored_contour = colorize_contour(&contour_pixels, &person_mask, source);

    Ok(YoloFrame {
        person_mask,
        contour,
        contour_pixels,
        colored_contour,
        confidence: Some(confidence),
    })
}

fn restore_mask(
    logits: Tensor,
    bounding_box: [f32; 4],
    transform: LetterboxTransform,
) -> Result<(GrayImage, MaskBounds)> {
    let model_mask = logits.upsample_bilinear2d(
        [MODEL_IMAGE_SIZE as i64, MODEL_IMAGE_SIZE as i64],
        false,
        None,
        None,
    );
    let unpadded = model_mask
        .narrow(2, transform.pad_y as i64, transform.resized_height as i64)
        .narrow(3, transform.pad_x as i64, transform.resized_width as i64);
    let restored = unpadded
        .upsample_bilinear2d(
            [
                transform.source_height as i64,
                transform.source_width as i64,
            ],
            false,
            None,
            None,
        )
        .gt(0.0)
        .to_kind(Kind::Uint8)
        * 255;
    let restored = restored.to_device(Device::Cpu).contiguous().view([-1]);
    let mut pixels = Vec::<u8>::try_from(&restored).context("copying YOLO mask to the CPU")?;

    let [center_x, center_y, width, height] = bounding_box;
    let left = ((center_x - width * 0.5 - transform.pad_x as f32) / transform.scale)
        .clamp(0.0, transform.source_width as f32);
    let right = ((center_x + width * 0.5 - transform.pad_x as f32) / transform.scale)
        .clamp(0.0, transform.source_width as f32);
    let top = ((center_y - height * 0.5 - transform.pad_y as f32) / transform.scale)
        .clamp(0.0, transform.source_height as f32);
    let bottom = ((center_y + height * 0.5 - transform.pad_y as f32) / transform.scale)
        .clamp(0.0, transform.source_height as f32);

    let width = transform.source_width as usize;
    let left = left.ceil() as usize;
    let right = right.ceil() as usize;
    let top = top.ceil() as usize;
    let bottom = bottom.ceil() as usize;
    for (y, row) in pixels.chunks_exact_mut(width).enumerate() {
        if y < top || y >= bottom {
            row.fill(0);
        } else {
            row[..left].fill(0);
            row[right..].fill(0);
        }
    }

    let mask = GrayImage::from_raw(transform.source_width, transform.source_height, pixels)
        .context("YOLO mask dimensions do not match its pixel buffer")?;
    Ok((
        mask,
        MaskBounds {
            left,
            right,
            top,
            bottom,
        },
    ))
}

fn extract_contour(mask: &GrayImage, bounds: MaskBounds) -> (GrayImage, Vec<usize>) {
    let width = mask.width() as usize;
    let height = mask.height() as usize;
    let source = mask.as_raw();
    let mut contour = vec![0; source.len()];
    let mut contour_pixels = Vec::new();

    for y in bounds.top..bounds.bottom {
        for x in bounds.left..bounds.right {
            let index = y * width + x;
            if source[index] != 0
                && (x == 0
                    || y == 0
                    || x + 1 == width
                    || y + 1 == height
                    || source[index - 1] == 0
                    || source[index + 1] == 0
                    || source[index - width] == 0
                    || source[index + width] == 0)
            {
                contour[index] = 255;
                contour_pixels.push(index);
            }
        }
    }

    (
        GrayImage::from_raw(mask.width(), mask.height(), contour)
            .expect("contour retains the source-mask dimensions"),
        contour_pixels,
    )
}

fn colorize_contour(
    contour_pixels: &[usize],
    person_mask: &GrayImage,
    source: &RgbaImage,
) -> RgbImage {
    debug_assert_eq!(person_mask.dimensions(), source.dimensions());

    let width = person_mask.width() as usize;
    let height = person_mask.height() as usize;
    let mut colors = vec![0; person_mask.as_raw().len() * 3];
    for &index in contour_pixels {
        let color = foreground_color(index, width, height, person_mask.as_raw(), source.as_raw());
        colors[index * 3..index * 3 + 3].copy_from_slice(&color);
    }

    RgbImage::from_raw(person_mask.width(), person_mask.height(), colors)
        .expect("person mask determines the color buffer dimensions")
}

fn foreground_color(
    index: usize,
    width: usize,
    height: usize,
    person_mask: &[u8],
    source: &[u8],
) -> [u8; 3] {
    let center = source_color(source, index);
    if !is_green_screen(center) {
        return center;
    }

    let x = index % width;
    let y = index / width;
    for radius in 1..=CONTOUR_COLOR_SEARCH_RADIUS {
        let mut best = None;
        for offset_y in -radius..=radius {
            for offset_x in -radius..=radius {
                if offset_x.abs() != radius && offset_y.abs() != radius {
                    continue;
                }

                let candidate_x = x as i32 + offset_x;
                let candidate_y = y as i32 + offset_y;
                if candidate_x < 0
                    || candidate_y < 0
                    || candidate_x >= width as i32
                    || candidate_y >= height as i32
                {
                    continue;
                }

                let candidate = candidate_y as usize * width + candidate_x as usize;
                if person_mask[candidate] == 0 {
                    continue;
                }

                let color = source_color(source, candidate);
                if is_green_screen(color) {
                    continue;
                }

                let score = laser_color_score(color);
                if best.is_none_or(|(_, best_score)| score > best_score) {
                    best = Some((color, score));
                }
            }
        }

        if let Some((color, _)) = best {
            return color;
        }
    }

    [0, 0, 0]
}

fn source_color(source: &[u8], pixel: usize) -> [u8; 3] {
    let offset = pixel * 4;
    [source[offset], source[offset + 1], source[offset + 2]]
}

fn is_green_screen([red, green, blue]: [u8; 3]) -> bool {
    green >= 80 && green.saturating_sub(red.max(blue)) >= 35
}

fn laser_color_score([red, green, blue]: [u8; 3]) -> u16 {
    let brightest = red.max(green).max(blue);
    let darkest = red.min(green).min(blue);
    brightest as u16 * 2 + (brightest - darkest) as u16
}

fn empty_frame(transform: LetterboxTransform) -> YoloFrame {
    YoloFrame {
        person_mask: GrayImage::new(transform.source_width, transform.source_height),
        contour: GrayImage::new(transform.source_width, transform.source_height),
        contour_pixels: Vec::new(),
        colored_contour: RgbImage::new(transform.source_width, transform.source_height),
        confidence: None,
    }
}

fn tensor_output(value: IValue, name: &str) -> Result<Tensor> {
    let IValue::Tensor(tensor) = value else {
        bail!("YOLO {name} output is not a tensor");
    };
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::{MaskBounds, colorize_contour, extract_contour};
    use image::{GrayImage, Luma, Rgba, RgbaImage};

    #[test]
    fn contour_retains_only_mask_boundary() {
        let mut mask = GrayImage::new(5, 5);
        for y in 1..=3 {
            for x in 1..=3 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }

        let (contour, contour_pixels) = extract_contour(
            &mask,
            MaskBounds {
                left: 0,
                right: mask.width() as usize,
                top: 0,
                bottom: mask.height() as usize,
            },
        );
        assert_eq!(contour.get_pixel(2, 2)[0], 0);
        assert_eq!(contour.get_pixel(1, 1)[0], 255);
        assert_eq!(contour.get_pixel(3, 2)[0], 255);
        assert_eq!(contour.pixels().filter(|pixel| pixel[0] > 0).count(), 8);
        assert_eq!(contour_pixels, [6, 7, 8, 11, 13, 16, 17, 18]);
    }

    #[test]
    fn colorized_contour_retains_source_colors_only_on_the_outline() {
        let source = RgbaImage::from_pixel(2, 1, Rgba([10, 20, 30, 255]));

        let person_mask = GrayImage::from_pixel(2, 1, Luma([255]));
        let colors = colorize_contour(&[1], &person_mask, &source);
        assert_eq!(colors.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(colors.get_pixel(1, 0).0, [10, 20, 30]);
    }

    #[test]
    fn colorized_contour_replaces_green_spill_with_nearby_foreground() {
        let person_mask = GrayImage::from_pixel(3, 1, Luma([255]));
        let mut source = RgbaImage::from_pixel(3, 1, Rgba([0, 180, 20, 255]));
        source.put_pixel(2, 0, Rgba([210, 90, 60, 255]));

        let colors = colorize_contour(&[1], &person_mask, &source);
        assert_eq!(colors.get_pixel(1, 0).0, [210, 90, 60]);
    }
}
