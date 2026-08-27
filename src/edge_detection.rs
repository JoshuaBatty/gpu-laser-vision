//! GPU-accelerated edge-detection pipeline.
//!
//! This module owns the CUDA setup and kernel execution, then reconstructs the
//! retained debug outputs as host-side images.

use crate::{cuda_graph::CapturedCudaGraph, kernels};
use anyhow::{Context, Result};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig, PinnedHostBuffer};
use cutile_cuda_async::error::DeviceError;
use image::{GrayImage, ImageReader, RgbImage, RgbaImage};
use std::{any::Any, fmt::Display, path::Path, sync::Arc};

pub const DEFAULT_MIN_THRESHOLD: f32 = 0.5;
pub const DEFAULT_MAX_THRESHOLD: f32 = 1.0;

/// Images produced by the GPU edge-detection pipeline.
pub struct EdgeDetectionImages {
    /// Original colour image supplied to the GPU pipeline.
    pub original: RgbaImage,
    /// Grayscale image used as input to the edge detector.
    pub grayscale: GrayImage,
    /// Gradient magnitude produced by the Scharr operator.
    pub edges: GrayImage,
    /// Binary laser mask thresholded directly from the Scharr magnitude.
    pub laser_edges: GrayImage,
    /// Original source colours sampled around each final edge on the GPU.
    pub edge_colors: RgbImage,
}

/// Persistent CUDA resources and captured graph for one frame size.
pub struct CudaEdgeDetector {
    // Drop the graph before the module, stream, and graph arguments.
    graph: CapturedCudaGraph,
    _module: Box<dyn Any>,
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    rgba_dev: DeviceBuffer<u8>,
    _grayscale_dev: DeviceBuffer<f32>,
    _edges_dev: DeviceBuffer<f32>,
    thresholds_dev: DeviceBuffer<f32>,
    grayscale_display_dev: DeviceBuffer<u8>,
    edges_display_dev: DeviceBuffer<u8>,
    _grad_x_dev: DeviceBuffer<f32>,
    _grad_y_dev: DeviceBuffer<f32>,
    edge_colors_dev: DeviceBuffer<u32>,
    grayscale_host: PinnedHostBuffer<u8>,
    edges_host: PinnedHostBuffer<u8>,
    edge_colors_host: PinnedHostBuffer<u32>,
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
    detector.process(&rgba, DEFAULT_MIN_THRESHOLD, DEFAULT_MAX_THRESHOLD)
}

impl CudaEdgeDetector {
    /// Allocates frame-sized resources and captures the CUDA graph.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;
        anyhow::ensure!(width > 0 && height > 0, "frame dimensions must be non-zero");

        // Initialize CUDA once for every sequence of equally sized frames.
        let ctx = CudaContext::new(0)?;

        let stream = ctx.new_stream()?;

        // Graph arguments remain at fixed addresses for the detector's lifetime.
        let rgba_dev = DeviceBuffer::<u8>::zeroed(&stream, n * 4)?;
        let mut grayscale_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let mut edges_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let thresholds_dev = DeviceBuffer::<f32>::zeroed(&stream, 2)?;
        let mut grad_x_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let mut grad_y_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let mut grayscale_display_dev = DeviceBuffer::<u8>::zeroed(&stream, n)?;
        let mut edges_display_dev = DeviceBuffer::<u8>::zeroed(&stream, n)?;
        let mut edge_colors_dev = DeviceBuffer::<u32>::zeroed(&stream, n)?;

        // Keep the module alive because captured kernel nodes reference its functions.
        let module = kernels::load(&ctx).context("loading embedded CUDA module")?;

        let graph = CapturedCudaGraph::capture(ctx.clone(), stream.clone(), || {
            unsafe {
                module.convert_to_grayscale(
                    &stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &rgba_dev,
                    &mut grayscale_dev,
                )
            }
            .capture_context("launching grayscale kernel")?;

            unsafe {
                module.scharr(
                    &stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &grayscale_dev,
                    &mut edges_dev,
                    &mut grad_x_dev,
                    &mut grad_y_dev,
                    w,
                    h,
                )
            }
            .capture_context("launching Scharr kernel")?;

            unsafe {
                module.extract_laser_edges(
                    &stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &rgba_dev,
                    &edges_dev,
                    &grad_x_dev,
                    &grad_y_dev,
                    &mut edge_colors_dev,
                    w,
                    &thresholds_dev,
                )
            }
            .capture_context("extracting laser edges and colours")?;

            unsafe {
                module.normalized_f32_to_u8(
                    &stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &grayscale_dev,
                    &mut grayscale_display_dev,
                )
            }
            .capture_context("converting grayscale output for display")?;

            unsafe {
                module.normalized_f32_to_u8(
                    &stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &edges_dev,
                    &mut edges_display_dev,
                )
            }
            .capture_context("converting Scharr output for display")?;
            Ok(())
        })?;

        let grayscale_host = PinnedHostBuffer::zeroed(&ctx, n)?;
        let edges_host = PinnedHostBuffer::zeroed(&ctx, n)?;
        let edge_colors_host = PinnedHostBuffer::zeroed(&ctx, n)?;

        Ok(Self {
            graph,
            _module: Box::new(module),
            _context: ctx,
            stream,
            rgba_dev,
            _grayscale_dev: grayscale_dev,
            _edges_dev: edges_dev,
            thresholds_dev,
            grayscale_display_dev,
            edges_display_dev,
            _grad_x_dev: grad_x_dev,
            _grad_y_dev: grad_y_dev,
            edge_colors_dev,
            grayscale_host,
            edges_host,
            edge_colors_host,
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
        rgba: &RgbaImage,
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
        // SAFETY: the following RGBA upload synchronizes this same stream before
        // `thresholds` can be dropped or reused.
        unsafe {
            self.thresholds_dev
                .copy_from_host_async_unchecked(&self.stream, &thresholds)?;
        }
        self.rgba_dev.copy_from_host(&self.stream, rgba.as_raw())?;
        self.graph.launch()?;

        self.grayscale_display_dev
            .copy_to_pinned_host(&self.stream, &mut self.grayscale_host)?;
        self.edges_display_dev
            .copy_to_pinned_host(&self.stream, &mut self.edges_host)?;
        self.edge_colors_dev
            .copy_to_pinned_host(&self.stream, &mut self.edge_colors_host)?;

        let mut laser_edges = Vec::with_capacity(self.edge_colors_host.len());
        let mut edge_colors = Vec::with_capacity(self.edge_colors_host.len() * 3);
        for color in self.edge_colors_host.iter().copied() {
            laser_edges.push((color >> 24) as u8);
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

        Ok(EdgeDetectionImages {
            grayscale: image(self.grayscale_host.to_vec())?,
            edges: image(self.edges_host.to_vec())?,
            laser_edges: image(laser_edges)?,
            edge_colors: RgbImage::from_raw(self.width, self.height, edge_colors).ok_or_else(
                || anyhow::anyhow!("edge-colour buffer size does not match image dimensions"),
            )?,
            original: rgba.clone(),
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
