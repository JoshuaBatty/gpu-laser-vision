//! GPU-accelerated edge-detection pipeline.
//!
//! This module owns the CUDA setup and kernel execution, then reconstructs the
//! retained debug outputs as host-side images.

use crate::{cuda_graph::CapturedCudaGraph, kernels};
use anyhow::{Context, Result};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig, PinnedHostBuffer};
use cutile_cuda_async::error::DeviceError;
use image::{GrayImage, ImageReader, RgbImage, RgbaImage};
use std::{fmt::Display, path::Path, sync::Arc};

/// Inclusive lower bound for normalized Scharr magnitudes.
pub const DEFAULT_MIN_THRESHOLD: f32 = 0.5;
/// Inclusive upper bound for normalized Scharr magnitudes.
pub const DEFAULT_MAX_THRESHOLD: f32 = 1.0;

/// Images produced by the GPU edge-detection pipeline.
pub struct EdgeDetectionImages {
    /// Display-ready images copied from intermediate CUDA buffers.
    pub previews: EdgeDetectionPreviews,
    /// Binary laser mask thresholded directly from the Scharr magnitude.
    pub laser_edges: GrayImage,
    /// Row-major indices of every non-zero pixel in `laser_edges`.
    pub edge_pixels: Vec<usize>,
    /// Original source colours sampled around each final edge on the GPU.
    pub edge_colors: RgbImage,
}

/// Host-side preview images copied from intermediate CUDA buffers.
pub struct EdgeDetectionPreviews {
    /// Grayscale image used as input to the edge detector.
    pub grayscale: GrayImage,
    /// Gradient magnitude produced by the Scharr operator.
    pub edges: GrayImage,
}

/// Device allocations and CUDA objects referenced by the captured graph.
struct EdgeGraphResources {
    module: kernels::LoadedModule,
    stream: Arc<CudaStream>,
    input_rgba: DeviceBuffer<u8>,
    grayscale: DeviceBuffer<f32>,
    scharr_magnitude: DeviceBuffer<f32>,
    thresholds: DeviceBuffer<f32>,
    gradient_x: DeviceBuffer<f32>,
    gradient_y: DeviceBuffer<f32>,
    grayscale_preview: DeviceBuffer<u8>,
    scharr_preview: DeviceBuffer<u8>,
    edge_colors: DeviceBuffer<u32>,
}

/// Pinned staging memory for results copied back from the GPU.
struct HostOutputs {
    grayscale: PinnedHostBuffer<u8>,
    scharr_magnitude: PinnedHostBuffer<u8>,
    edge_colors: PinnedHostBuffer<u32>,
}

/// Persistent CUDA resources and captured graph for one frame size.
pub struct CudaEdgeDetector {
    graph: CapturedCudaGraph<EdgeGraphResources>,
    host: HostOutputs,
    last_thresholds: Option<[f32; 2]>,
    width: u32,
    height: u32,
}

/// Runs the GPU edge-detection pipeline on the image at `path`.
///
/// Returns the retained outputs, or an error from image loading, CUDA setup
/// and execution, or reconstruction of the output images.
pub fn process(path: impl AsRef<Path>) -> Result<EdgeDetectionImages> {
    let path = path.as_ref();
    let rgba = ImageReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .decode()?
        .into_rgba8();

    process_rgba(rgba)
}

/// Runs the GPU edge-detection pipeline on an in-memory RGBA image.
pub fn process_rgba(rgba: RgbaImage) -> Result<EdgeDetectionImages> {
    let mut detector = CudaEdgeDetector::new(rgba.width(), rgba.height())?;
    detector.process(rgba, DEFAULT_MIN_THRESHOLD, DEFAULT_MAX_THRESHOLD)
}

impl CudaEdgeDetector {
    /// Allocates frame-sized resources and captures the CUDA graph.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;
        anyhow::ensure!(width > 0 && height > 0, "frame dimensions must be non-zero");

        // Initialize CUDA once for every sequence of equally sized frames.
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;
        let resources = EdgeGraphResources {
            module: kernels::load(&context).context("loading embedded CUDA module")?,
            stream: Arc::clone(&stream),
            input_rgba: DeviceBuffer::zeroed(&stream, n * 4)?,
            grayscale: DeviceBuffer::zeroed(&stream, n)?,
            scharr_magnitude: DeviceBuffer::zeroed(&stream, n)?,
            thresholds: DeviceBuffer::zeroed(&stream, 2)?,
            gradient_x: DeviceBuffer::zeroed(&stream, n)?,
            gradient_y: DeviceBuffer::zeroed(&stream, n)?,
            grayscale_preview: DeviceBuffer::zeroed(&stream, n)?,
            scharr_preview: DeviceBuffer::zeroed(&stream, n)?,
            edge_colors: DeviceBuffer::zeroed(&stream, n)?,
        };
        let host = HostOutputs {
            grayscale: PinnedHostBuffer::zeroed(&context, n)?,
            scharr_magnitude: PinnedHostBuffer::zeroed(&context, n)?,
            edge_colors: PinnedHostBuffer::zeroed(&context, n)?,
        };

        let graph = CapturedCudaGraph::capture(context, stream, resources, |resources| {
            unsafe {
                resources.module.convert_to_grayscale(
                    &resources.stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &resources.input_rgba,
                    &mut resources.grayscale,
                )
            }
            .capture_context("launching grayscale kernel")?;

            unsafe {
                resources.module.scharr(
                    &resources.stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &resources.grayscale,
                    &mut resources.scharr_magnitude,
                    &mut resources.gradient_x,
                    &mut resources.gradient_y,
                    w,
                    h,
                )
            }
            .capture_context("launching Scharr kernel")?;

            unsafe {
                resources.module.extract_laser_edges(
                    &resources.stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &resources.input_rgba,
                    &resources.scharr_magnitude,
                    &resources.gradient_x,
                    &resources.gradient_y,
                    &mut resources.edge_colors,
                    w,
                    &resources.thresholds,
                )
            }
            .capture_context("extracting laser edges and colours")?;

            unsafe {
                resources.module.normalized_f32_to_u8(
                    &resources.stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &resources.grayscale,
                    &mut resources.grayscale_preview,
                )
            }
            .capture_context("converting grayscale output for display")?;

            unsafe {
                resources.module.normalized_f32_to_u8(
                    &resources.stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &resources.scharr_magnitude,
                    &mut resources.scharr_preview,
                )
            }
            .capture_context("converting Scharr output for display")?;
            Ok(())
        })?;

        Ok(Self {
            graph,
            host,
            last_thresholds: None,
            width,
            height,
        })
    }

    /// Returns the frame dimensions accepted by this captured graph.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Processes one frame and copies the retained outputs back to the host.
    pub fn process(
        &mut self,
        rgba: RgbaImage,
        min_threshold: f32,
        max_threshold: f32,
    ) -> Result<EdgeDetectionImages> {
        anyhow::ensure!(
            rgba.dimensions() == self.dimensions(),
            "frame dimensions changed from {}x{} to {}x{}",
            self.width,
            self.height,
            rgba.width(),
            rgba.height()
        );
        anyhow::ensure!(
            min_threshold.is_finite()
                && max_threshold.is_finite()
                && 0.0 <= min_threshold
                && min_threshold <= max_threshold
                && max_threshold <= 1.0,
            "edge thresholds must satisfy 0 <= min <= max <= 1"
        );

        let thresholds = [min_threshold, max_threshold];
        {
            let resources = self.graph.resources_mut();
            if self.last_thresholds != Some(thresholds) {
                // SAFETY: the following RGBA upload synchronizes this same
                // stream before `thresholds` can be dropped or reused.
                unsafe {
                    resources
                        .thresholds
                        .copy_from_host_async_unchecked(&resources.stream, &thresholds)?;
                }
                self.last_thresholds = Some(thresholds);
            }
            resources
                .input_rgba
                .copy_from_host(&resources.stream, rgba.as_raw())?;
        }
        self.graph.launch()?;

        let resources = self.graph.resources_mut();
        resources
            .grayscale_preview
            .copy_to_pinned_host(&resources.stream, &mut self.host.grayscale)?;
        resources
            .scharr_preview
            .copy_to_pinned_host(&resources.stream, &mut self.host.scharr_magnitude)?;
        resources
            .edge_colors
            .copy_to_pinned_host(&resources.stream, &mut self.host.edge_colors)?;

        let mut laser_edges = Vec::with_capacity(self.host.edge_colors.len());
        let mut edge_pixels = Vec::new();
        let mut edge_colors = Vec::with_capacity(self.host.edge_colors.len() * 3);
        for (pixel, color) in self.host.edge_colors.iter().copied().enumerate() {
            let edge = (color >> 24) as u8;
            laser_edges.push(edge);
            if edge != 0 {
                edge_pixels.push(pixel);
            }
            edge_colors.extend_from_slice(&[
                (color & 0xff) as u8,
                ((color >> 8) & 0xff) as u8,
                ((color >> 16) & 0xff) as u8,
            ]);
        }

        let image = |data| {
            GrayImage::from_raw(self.width, self.height, data).ok_or_else(|| {
                anyhow::anyhow!("grayscale buffer size does not match image dimensions")
            })
        };
        let previews = EdgeDetectionPreviews {
            grayscale: image(self.host.grayscale.to_vec())?,
            edges: image(self.host.scharr_magnitude.to_vec())?,
        };

        Ok(EdgeDetectionImages {
            previews,
            laser_edges: image(laser_edges)?,
            edge_pixels,
            edge_colors: RgbImage::from_raw(self.width, self.height, edge_colors).ok_or_else(
                || anyhow::anyhow!("edge-colour buffer size does not match image dimensions"),
            )?,
        })
    }
}

trait CaptureResultExt<T> {
    fn capture_context(self, operation: &'static str) -> std::result::Result<T, DeviceError>;
}

impl<T, E: Display> CaptureResultExt<T> for std::result::Result<T, E> {
    fn capture_context(self, operation: &'static str) -> std::result::Result<T, DeviceError> {
        self.map_err(|error| DeviceError::Scheduling(format!("{operation}: {error}")))
    }
}
