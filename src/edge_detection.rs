use crate::kernels;
use anyhow::{Context, Result};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use image::{GrayImage, ImageReader};
use std::path::Path;

pub struct EdgeDetectionImages {
    pub grayscale: GrayImage,
    pub edges: GrayImage,
    pub thin_edges: GrayImage,
    pub edge_classes: GrayImage,
    pub connected_edges: GrayImage,
}

pub fn process(path: impl AsRef<Path>) -> Result<EdgeDetectionImages> {
    // Initialize CUDA
    let ctx = CudaContext::new(0)?;

    // Init CUDA streams
    let grayscale_stream = ctx.new_stream()?;
    let edge_stream = ctx.new_stream()?;
    let nms_stream = ctx.new_stream()?;
    let threshold_stream = ctx.new_stream()?;
    let hysteresis_stream = ctx.new_stream()?;

    // Load an image
    let path = path.as_ref();
    let img = ImageReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .decode()?;

    let w = img.width() as usize;
    let h = img.height() as usize;
    let n = w * h;
    let rgba = img.into_rgba8();

    // Allocate device memory
    let rgba_dev = DeviceBuffer::from_host(&grayscale_stream, rgba.as_raw())?;
    let mut grayscale_dev = DeviceBuffer::<f32>::zeroed(&grayscale_stream, n)?;
    let mut edges_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
    let mut thin_edges_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
    let mut edge_classes_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
    let mut connected_edges_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
    let mut connected_edges_next_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
    let mut grad_x_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;
    let mut grad_y_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, n)?;

    // Load embedded PTX module, exposes generated kernel launchers.
    let module = kernels::load(&ctx).context("loading embedded CUDA module")?;

    // Launch grayscale conversion kernel.
    unsafe {
        module.convert_to_grayscale(
            &grayscale_stream,
            LaunchConfig::for_num_elems(n as u32),
            &rgba_dev,
            &mut grayscale_dev,
        )
    }
    .context("launching grayscale kernel")?;

    // Record when the grayscale kernal has completed its work.
    let grayscale_done = grayscale_stream.record_event(None)?;

    // Do not run edge detection until grayscale is complete.
    edge_stream.wait(&grayscale_done)?;

    // Launch the Scharr edge detection kernel.
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
    .context("launching Scharr kernel")?;

    // Record when the edge kernal has completed its work.
    let edge_done = edge_stream.record_event(None)?;

    // Do not run nms until edge is complete.
    nms_stream.wait(&edge_done)?;

    // Launch the nms kernel.
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
    .context("launching non-maximum-suppression kernel")?;

    // Record when the nms kernal has completed its work.
    let nms_done = nms_stream.record_event(None)?;

    // Do not run nms until edge is complete.
    threshold_stream.wait(&nms_done)?;

    let low_threshold: f32 = 0.022;
    let high_threshold: f32 = 0.045;

    // Launch the threshold kernel.
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
    .context("launching double-threshold kernel")?;

    // Record when the threshold kernal has completed its work.
    let threshold_done = threshold_stream.record_event(None)?;

    // Do not run nms until threshold is complete.
    hysteresis_stream.wait(&threshold_done)?;

    // Launch the hysteresis kernel.
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
        .context("launching hysteresis kernel")?;

        std::mem::swap(&mut connected_edges_dev, &mut connected_edges_next_dev);
    }

    // Get the results
    let grayscale_host = grayscale_dev.to_host_vec(&grayscale_stream)?;
    let edges_host = edges_dev.to_host_vec(&edge_stream)?;
    let thin_edges_host = thin_edges_dev.to_host_vec(&nms_stream)?;
    let edge_classes_host = edge_classes_dev.to_host_vec(&threshold_stream)?;
    let connected_edges_host = connected_edges_dev.to_host_vec(&hysteresis_stream)?;

    println!(
        "Thin-edge max: {}",
        thin_edges_host.iter().copied().fold(0.0_f32, f32::max)
    );

    // Convert float output to grayscale bytes and save png
    let grayscale: Vec<u8> = normalized_f32_to_u8(&grayscale_host, 1.0);
    let edges: Vec<u8> = normalized_f32_to_u8(&edges_host, 1.0);
    let thin_edges: Vec<u8> = normalized_f32_to_u8(&thin_edges_host, 12.0);
    let edge_classes: Vec<u8> = normalized_f32_to_u8(&edge_classes_host, 1.0);
    let connected_edges: Vec<u8> = normalized_f32_to_u8(&connected_edges_host, 1.0);

    let image = |data| {
        GrayImage::from_raw(rgba.width(), rgba.height(), data)
            .ok_or_else(|| anyhow::anyhow!("grayscale buffer size does not match image dimensions"))
    };

    Ok(EdgeDetectionImages {
        grayscale: image(grayscale)?,
        edges: image(edges)?,
        thin_edges: image(thin_edges)?,
        edge_classes: image(edge_classes)?,
        connected_edges: image(connected_edges)?,
    })
}

fn normalized_f32_to_u8(values: &[f32], gain: f32) -> Vec<u8> {
    values
        .iter()
        .map(|&value| ((value * gain).clamp(0.0, 1.0) * 255.0) as u8)
        .collect()
}
