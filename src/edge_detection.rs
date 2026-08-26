//! GPU-accelerated edge-detection pipeline.
//!
//! This module owns the CUDA setup and kernel execution, then reconstructs
//! each processing stage as a host-side image.

use crate::{cuda_graph::CapturedCudaGraph, kernels};
use anyhow::{Context, Result};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig, PinnedHostBuffer};
use cutile_cuda_async::error::DeviceError;
use image::{GrayImage, ImageReader, RgbImage, RgbaImage};
use std::{any::Any, fmt::Display, path::Path, sync::Arc};

const DEFAULT_LASER_EDGE_THRESHOLD: f32 = 0.5;
const LASER_EDGE_THRESHOLD_ENV: &str = "LASER_EDGE_THRESHOLD";

/// Images produced by each stage of the GPU edge-detection pipeline.
pub struct EdgeDetectionImages {
    /// Original colour image supplied to the GPU pipeline.
    pub original: RgbaImage,
    /// Grayscale image used as input to the edge detector.
    pub grayscale: GrayImage,
    /// Gradient magnitude produced by the Scharr operator.
    pub edges: GrayImage,
    /// Edges thinned by non-maximum suppression.
    pub thin_edges: GrayImage,
    /// Weak and strong edge classes produced by double thresholding.
    pub edge_classes: GrayImage,
    /// Final edges retained after hysteresis connectivity tracking.
    pub connected_edges: GrayImage,
    /// Binary laser mask thresholded directly from the Scharr magnitude.
    pub laser_edges: GrayImage,
    /// Original source colours sampled around each final edge on the GPU.
    pub edge_colors: RgbImage,
}

/// Persistent CUDA resources and captured graph for one frame size.
pub struct CudaEdgeDetector {
    // Drop the graph before the module, events, streams, and graph arguments.
    graph: CapturedCudaGraph,
    _module: Box<dyn Any>,
    _events: Vec<Box<dyn Any>>,
    _context: Arc<CudaContext>,
    grayscale_stream: Arc<CudaStream>,
    _edge_stream: Arc<CudaStream>,
    _nms_stream: Arc<CudaStream>,
    _threshold_stream: Arc<CudaStream>,
    hysteresis_stream: Arc<CudaStream>,
    rgba_dev: DeviceBuffer<u8>,
    _grayscale_dev: DeviceBuffer<f32>,
    _edges_dev: DeviceBuffer<f32>,
    _thin_edges_dev: DeviceBuffer<f32>,
    _edge_classes_dev: DeviceBuffer<f32>,
    _connected_edges_dev: DeviceBuffer<f32>,
    _laser_edges_dev: DeviceBuffer<f32>,
    grayscale_display_dev: DeviceBuffer<u8>,
    edges_display_dev: DeviceBuffer<u8>,
    thin_edges_display_dev: DeviceBuffer<u8>,
    edge_classes_display_dev: DeviceBuffer<u8>,
    connected_edges_display_dev: DeviceBuffer<u8>,
    laser_edges_display_dev: DeviceBuffer<u8>,
    _grad_x_dev: DeviceBuffer<f32>,
    _grad_y_dev: DeviceBuffer<f32>,
    edge_colors_dev: DeviceBuffer<u32>,
    grayscale_host: PinnedHostBuffer<u8>,
    edges_host: PinnedHostBuffer<u8>,
    thin_edges_host: PinnedHostBuffer<u8>,
    edge_classes_host: PinnedHostBuffer<u8>,
    connected_edges_host: PinnedHostBuffer<u8>,
    laser_edges_host: PinnedHostBuffer<u8>,
    edge_colors_host: PinnedHostBuffer<u32>,
    width: u32,
    height: u32,
}

/// Runs the GPU edge-detection pipeline on the image at `path`.
///
/// Returns the output of every stage, or an error from image loading, CUDA
/// setup and execution, or reconstruction of the output images.
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
    detector.process(&rgba)
}

impl CudaEdgeDetector {
    /// Allocates frame-sized resources and captures the five-stream CUDA graph.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;
        let laser_edge_threshold = std::env::var(LASER_EDGE_THRESHOLD_ENV)
            .ok()
            .map(|value| {
                value
                    .parse::<f32>()
                    .with_context(|| format!("parsing {LASER_EDGE_THRESHOLD_ENV}={value}"))
            })
            .transpose()?
            .unwrap_or(DEFAULT_LASER_EDGE_THRESHOLD);
        anyhow::ensure!(
            laser_edge_threshold.is_finite() && laser_edge_threshold >= 0.0,
            "{LASER_EDGE_THRESHOLD_ENV} must be a finite non-negative number"
        );

        // Initialize CUDA once for every sequence of equally sized frames.
        let ctx = CudaContext::new(0)?;

        let grayscale_stream = ctx.new_stream()?;
        let edge_stream = ctx.new_stream()?;
        let nms_stream = ctx.new_stream()?;
        let threshold_stream = ctx.new_stream()?;
        let hysteresis_stream = ctx.new_stream()?;

        // Graph arguments remain at fixed addresses for the detector's lifetime.
        let rgba_dev = DeviceBuffer::<u8>::zeroed(&grayscale_stream, n * 4)?;
        let mut grayscale_dev = DeviceBuffer::<f32>::zeroed(&grayscale_stream, n)?;
        let mut edges_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
        let mut thin_edges_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
        let mut edge_classes_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
        let mut connected_edges_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
        let mut connected_edges_next_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
        let mut grad_x_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
        let mut grad_y_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
        let mut grayscale_display_dev = DeviceBuffer::<u8>::zeroed(&hysteresis_stream, n)?;
        let mut edges_display_dev = DeviceBuffer::<u8>::zeroed(&hysteresis_stream, n)?;
        let mut thin_edges_display_dev = DeviceBuffer::<u8>::zeroed(&hysteresis_stream, n)?;
        let mut edge_classes_display_dev = DeviceBuffer::<u8>::zeroed(&hysteresis_stream, n)?;
        let mut connected_edges_display_dev = DeviceBuffer::<u8>::zeroed(&hysteresis_stream, n)?;
        let mut laser_edges_display_dev = DeviceBuffer::<u8>::zeroed(&hysteresis_stream, n)?;
        let mut edge_colors_dev = DeviceBuffer::<u32>::zeroed(&hysteresis_stream, n)?;

        // Keep the module alive because captured kernel nodes reference its functions.
        let module = kernels::load(&ctx).context("loading embedded CUDA module")?;
        let mut events: Vec<Box<dyn Any>> = Vec::with_capacity(5);

        let graph = CapturedCudaGraph::capture(ctx.clone(), grayscale_stream.clone(), || {
            unsafe {
                module.convert_to_grayscale(
                    &grayscale_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &rgba_dev,
                    &mut grayscale_dev,
                )
            }
            .capture_context("launching grayscale kernel")?;

            let grayscale_done = grayscale_stream
                .record_event(None)
                .capture_context("recording grayscale completion")?;
            edge_stream
                .wait(&grayscale_done)
                .capture_context("waiting for grayscale completion")?;
            events.push(Box::new(grayscale_done));

            unsafe {
                module.scharr(
                    &edge_stream,
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

            let edge_done = edge_stream
                .record_event(None)
                .capture_context("recording Scharr completion")?;
            nms_stream
                .wait(&edge_done)
                .capture_context("waiting for Scharr completion")?;
            events.push(Box::new(edge_done));

            unsafe {
                module.non_maximum_suppression(
                    &nms_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &edges_dev,
                    &grad_x_dev,
                    &grad_y_dev,
                    &mut thin_edges_dev,
                    w,
                    h,
                )
            }
            .capture_context("launching non-maximum-suppression kernel")?;

            let nms_done = nms_stream
                .record_event(None)
                .capture_context("recording non-maximum-suppression completion")?;
            threshold_stream
                .wait(&nms_done)
                .capture_context("waiting for non-maximum-suppression completion")?;
            events.push(Box::new(nms_done));

            let low_threshold: f32 = 0.022;
            let high_threshold: f32 = 0.045;

            unsafe {
                module.double_threshold(
                    &threshold_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &thin_edges_dev,
                    &mut edge_classes_dev,
                    low_threshold,
                    high_threshold,
                )
            }
            .capture_context("launching double-threshold kernel")?;

            let threshold_done = threshold_stream
                .record_event(None)
                .capture_context("recording double-threshold completion")?;
            hysteresis_stream
                .wait(&threshold_done)
                .capture_context("waiting for double-threshold completion")?;
            events.push(Box::new(threshold_done));

            connected_edges_dev
                .zero_async(&hysteresis_stream)
                .capture_context("clearing hysteresis buffer")?;
            connected_edges_next_dev
                .zero_async(&hysteresis_stream)
                .capture_context("clearing hysteresis staging buffer")?;

            for _ in 0..64 {
                unsafe {
                    module.hysteresis(
                        &hysteresis_stream,
                        LaunchConfig::for_num_elems(n as u32),
                        &edge_classes_dev,
                        &connected_edges_dev,
                        &mut connected_edges_next_dev,
                        w,
                        h,
                    )
                }
                .capture_context("launching hysteresis kernel")?;

                std::mem::swap(&mut connected_edges_dev, &mut connected_edges_next_dev);
            }

            // Build the laser mask directly from strong Scharr magnitudes. Equal
            // thresholds make the existing classifier emit only zero or one.
            unsafe {
                module.double_threshold(
                    &hysteresis_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &edges_dev,
                    &mut connected_edges_next_dev,
                    laser_edge_threshold,
                    laser_edge_threshold,
                )
            }
            .capture_context("thresholding Scharr magnitude for laser output")?;

            unsafe {
                module.colorize_edges(
                    &hysteresis_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &rgba_dev,
                    &connected_edges_next_dev,
                    &grad_x_dev,
                    &grad_y_dev,
                    &mut edge_colors_dev,
                    w,
                    h,
                    2,
                )
            }
            .capture_context("launching edge-colour recovery kernel")?;

            unsafe {
                module.normalized_f32_to_u8(
                    &hysteresis_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &grayscale_dev,
                    &mut grayscale_display_dev,
                    1.0,
                )
            }
            .capture_context("converting grayscale output for display")?;

            unsafe {
                module.normalized_f32_to_u8(
                    &hysteresis_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &edges_dev,
                    &mut edges_display_dev,
                    1.0,
                )
            }
            .capture_context("converting Scharr output for display")?;

            unsafe {
                module.normalized_f32_to_u8(
                    &hysteresis_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &thin_edges_dev,
                    &mut thin_edges_display_dev,
                    12.0,
                )
            }
            .capture_context("converting non-maximum-suppression output for display")?;

            unsafe {
                module.normalized_f32_to_u8(
                    &hysteresis_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &edge_classes_dev,
                    &mut edge_classes_display_dev,
                    1.0,
                )
            }
            .capture_context("converting threshold output for display")?;

            unsafe {
                module.normalized_f32_to_u8(
                    &hysteresis_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &connected_edges_dev,
                    &mut connected_edges_display_dev,
                    1.0,
                )
            }
            .capture_context("converting hysteresis output for display")?;

            unsafe {
                module.normalized_f32_to_u8(
                    &hysteresis_stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &connected_edges_next_dev,
                    &mut laser_edges_display_dev,
                    1.0,
                )
            }
            .capture_context("converting laser mask for display")?;

            let pipeline_done = hysteresis_stream
                .record_event(None)
                .capture_context("recording pipeline completion")?;
            grayscale_stream
                .wait(&pipeline_done)
                .capture_context("joining captured streams")?;
            events.push(Box::new(pipeline_done));
            Ok(())
        })?;

        let grayscale_host = PinnedHostBuffer::zeroed(&ctx, n)?;
        let edges_host = PinnedHostBuffer::zeroed(&ctx, n)?;
        let thin_edges_host = PinnedHostBuffer::zeroed(&ctx, n)?;
        let edge_classes_host = PinnedHostBuffer::zeroed(&ctx, n)?;
        let connected_edges_host = PinnedHostBuffer::zeroed(&ctx, n)?;
        let laser_edges_host = PinnedHostBuffer::zeroed(&ctx, n)?;
        let edge_colors_host = PinnedHostBuffer::zeroed(&ctx, n)?;

        Ok(Self {
            graph,
            _module: Box::new(module),
            _events: events,
            _context: ctx,
            grayscale_stream,
            _edge_stream: edge_stream,
            _nms_stream: nms_stream,
            _threshold_stream: threshold_stream,
            hysteresis_stream,
            rgba_dev,
            _grayscale_dev: grayscale_dev,
            _edges_dev: edges_dev,
            _thin_edges_dev: thin_edges_dev,
            _edge_classes_dev: edge_classes_dev,
            _connected_edges_dev: connected_edges_dev,
            _laser_edges_dev: connected_edges_next_dev,
            grayscale_display_dev,
            edges_display_dev,
            thin_edges_display_dev,
            edge_classes_display_dev,
            connected_edges_display_dev,
            laser_edges_display_dev,
            _grad_x_dev: grad_x_dev,
            _grad_y_dev: grad_y_dev,
            edge_colors_dev,
            grayscale_host,
            edges_host,
            thin_edges_host,
            edge_classes_host,
            connected_edges_host,
            laser_edges_host,
            edge_colors_host,
            width,
            height,
        })
    }

    /// Returns the frame dimensions accepted by this captured graph.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Processes one frame and copies every debug stage back to the host.
    pub fn process(&mut self, rgba: &RgbaImage) -> Result<EdgeDetectionImages> {
        anyhow::ensure!(
            rgba.dimensions() == self.dimensions(),
            "frame dimensions changed from {}x{} to {}x{}",
            self.width,
            self.height,
            rgba.width(),
            rgba.height()
        );

        self.rgba_dev
            .copy_from_host(&self.grayscale_stream, rgba.as_raw())?;
        self.graph.launch()?;

        self.grayscale_display_dev
            .copy_to_pinned_host(&self.hysteresis_stream, &mut self.grayscale_host)?;
        self.edges_display_dev
            .copy_to_pinned_host(&self.hysteresis_stream, &mut self.edges_host)?;
        self.thin_edges_display_dev
            .copy_to_pinned_host(&self.hysteresis_stream, &mut self.thin_edges_host)?;
        self.edge_classes_display_dev
            .copy_to_pinned_host(&self.hysteresis_stream, &mut self.edge_classes_host)?;
        self.connected_edges_display_dev
            .copy_to_pinned_host(&self.hysteresis_stream, &mut self.connected_edges_host)?;
        self.laser_edges_display_dev
            .copy_to_pinned_host(&self.hysteresis_stream, &mut self.laser_edges_host)?;
        self.edge_colors_dev
            .copy_to_pinned_host(&self.hysteresis_stream, &mut self.edge_colors_host)?;

        let edge_colors: Vec<u8> = self
            .edge_colors_host
            .iter()
            .copied()
            .flat_map(|color| {
                [
                    (color & 0xff) as u8,
                    ((color >> 8) & 0xff) as u8,
                    ((color >> 16) & 0xff) as u8,
                ]
            })
            .collect();

        let image = |data| {
            GrayImage::from_raw(self.width, self.height, data).ok_or_else(|| {
                anyhow::anyhow!("grayscale buffer size does not match image dimensions")
            })
        };

        Ok(EdgeDetectionImages {
            grayscale: image(self.grayscale_host.to_vec())?,
            edges: image(self.edges_host.to_vec())?,
            thin_edges: image(self.thin_edges_host.to_vec())?,
            edge_classes: image(self.edge_classes_host.to_vec())?,
            connected_edges: image(self.connected_edges_host.to_vec())?,
            laser_edges: image(self.laser_edges_host.to_vec())?,
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
