//! PyTorch-backed YOLO instance-segmentation inference.
//!
//! This module owns the LibTorch model, prepares fixed-size letterboxed input,
//! and exposes the raw prediction and mask-prototype tensors for post-processing.

use anyhow::{Context, Result, bail};
use image::{Rgb, RgbImage, RgbaImage, imageops};
use std::path::Path;
use tch::{CModule, Device, IValue, Kind, Tensor};

const MODEL_IMAGE_SIZE: u32 = 640;

/// Default TorchScript artifact produced by `scripts/export_yolo.py`.
pub const DEFAULT_MODEL_PATH: &str = "assets/yolo11n-seg.torchscript";

/// Geometry required to map model-space masks back into the source frame.
pub struct LetterboxTransform {
    /// Uniform source-to-model scale.
    pub scale: f32,
    /// Horizontal padding in model pixels.
    pub pad_x: u32,
    /// Vertical padding in model pixels.
    pub pad_y: u32,
}

/// Raw outputs from a YOLO instance-segmentation forward pass.
pub struct YoloSegmentation {
    /// Bounding boxes, class scores, and mask coefficients.
    pub predictions: Tensor,
    /// Prototype masks combined with per-detection mask coefficients.
    pub prototypes: Tensor,
    /// Transform used to prepare this inference input.
    pub transform: LetterboxTransform,
}

/// Loaded YOLO TorchScript model and its inference device.
pub struct YoloSegmenter {
    model: CModule,
    device: Device,
}

impl YoloSegmenter {
    /// Loads a TorchScript segmentation model onto CUDA when available.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_file() {
            bail!(
                "YOLO model not found at {}; generate it with `uv run scripts/export_yolo.py`",
                path.display()
            );
        }
        let device = Device::cuda_if_available();
        let mut model = CModule::load_on_device(path, device)
            .with_context(|| format!("loading YOLO model from {}", path.display()))?;
        model.set_eval();

        println!("YOLO inference device: {device:?}");
        Ok(Self { model, device })
    }

    /// Runs one inference pass and returns raw segmentation tensors.
    pub fn infer(&self, rgba: &RgbaImage) -> Result<YoloSegmentation> {
        let (input, transform) = prepare_input(rgba, self.device)?;
        let output = tch::no_grad(|| self.model.forward_is(&[IValue::Tensor(input)]))
            .context("running YOLO TorchScript inference")?;
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
        let predictions =
            tensor_output(values.pop().expect("output length checked"), "predictions")?;

        Ok(YoloSegmentation {
            predictions,
            prototypes,
            transform,
        })
    }
}

fn prepare_input(rgba: &RgbaImage, device: Device) -> Result<(Tensor, LetterboxTransform)> {
    let (width, height) = rgba.dimensions();
    anyhow::ensure!(width > 0 && height > 0, "YOLO input image is empty");

    let scale =
        (MODEL_IMAGE_SIZE as f32 / width as f32).min(MODEL_IMAGE_SIZE as f32 / height as f32);
    let resized_width = (width as f32 * scale).round() as u32;
    let resized_height = (height as f32 * scale).round() as u32;
    let pad_x = (MODEL_IMAGE_SIZE - resized_width) / 2;
    let pad_y = (MODEL_IMAGE_SIZE - resized_height) / 2;

    let rgb = image::DynamicImage::ImageRgba8(rgba.clone()).into_rgb8();
    let resized = imageops::resize(
        &rgb,
        resized_width,
        resized_height,
        imageops::FilterType::Triangle,
    );
    let mut letterboxed =
        RgbImage::from_pixel(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, Rgb([114, 114, 114]));
    imageops::replace(&mut letterboxed, &resized, pad_x.into(), pad_y.into());

    let input = Tensor::f_from_slice(letterboxed.as_raw())
        .context("creating YOLO input tensor")?
        .view([MODEL_IMAGE_SIZE as i64, MODEL_IMAGE_SIZE as i64, 3])
        .permute([2, 0, 1])
        .unsqueeze(0)
        .to_device(device)
        .to_kind(Kind::Float)
        / 255.0;

    Ok((
        input,
        LetterboxTransform {
            scale,
            pad_x,
            pad_y,
        },
    ))
}

fn tensor_output(value: IValue, name: &str) -> Result<Tensor> {
    let IValue::Tensor(tensor) = value else {
        bail!("YOLO {name} output is not a tensor");
    };
    Ok(tensor)
}
