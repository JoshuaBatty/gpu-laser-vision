mod kernels;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use image::ImageReader;

fn main() {
    // Initialize CUDA
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    // Init CUDA streams
    let grayscale_stream = ctx.new_stream().expect("Failed to create grayscale stream");
    let edge_stream = ctx.new_stream().expect("Failed to create edge stream");
    let nms_stream = ctx.new_stream().expect("Failed to create nms stream");
    let threshold_stream = ctx.new_stream().expect("Failed to create threshold stream");
    let hysteresis_stream = ctx
        .new_stream()
        .expect("Failed to create hysteresis stream");

    // Load an image
    let img = ImageReader::open("assets/test_tiles.png")
        .expect("unable to load image")
        .decode()
        .expect("unable to decode image");

    let w = img.width() as usize;
    let h = img.height() as usize;
    let N = w * h;
    let rgba = img.into_rgba8();

    // Allocate device memory
    let rgba_dev = DeviceBuffer::from_host(&grayscale_stream, rgba.as_raw()).unwrap();
    let mut grayscale_dev = DeviceBuffer::<f32>::zeroed(&grayscale_stream, N).unwrap();
    let mut edges_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, N).unwrap();
    let mut thin_edges_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, N).unwrap();
    let mut edge_classes_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, N).unwrap();
    let mut connected_edges_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, N).unwrap();
    let mut connected_edges_next_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, N).unwrap();

    let mut grad_x_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, N).unwrap();
    let mut grad_y_dev = DeviceBuffer::<f32>::zeroed(&edge_stream, N).unwrap();

    // Load embedded PTX module, exposes generated kernel launchers.
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // Launch grayscale conversion kernel.
    unsafe {
        module.convert_to_grayscale(
            &grayscale_stream,
            LaunchConfig::for_num_elems(N as u32),
            &rgba_dev,
            &mut grayscale_dev,
        )
    }
    .expect("Grayscale kernal launch failed");

    // Record when the grayscale kernal has completed its work.
    let grayscale_done = grayscale_stream
        .record_event(None)
        .expect("Failed to record grayscale event");

    // Do not run edge detection until grayscale is complete.
    edge_stream
        .wait(&grayscale_done)
        .expect("Failed to wait for grayscale event");

    // Launch the edge detection kernel.
    unsafe {
        module.edge_detect(
            &edge_stream,
            LaunchConfig::for_num_elems(N as u32),
            &grayscale_dev,
            &mut edges_dev,
            &mut grad_x_dev,
            &mut grad_y_dev,
            w,
            h,
        )
    }
    .expect("Edge detect kernal launch failed");

    // Record when the edge kernal has completed its work.
    let edge_done = edge_stream
        .record_event(None)
        .expect("Failed to record edge event");

    // Do not run nms until edge is complete.
    nms_stream
        .wait(&edge_done)
        .expect("Failed to wait for edge event");

    // Launch the nms kernel.
    unsafe {
        module.non_maximum_suppression(
            &nms_stream,
            LaunchConfig::for_num_elems(N as u32),
            &edges_dev,
            &grad_x_dev,
            &grad_y_dev,
            &mut thin_edges_dev,
            w,
            h,
        )
    }
    .expect("Edge detect kernal launch failed");

    // Record when the nms kernal has completed its work.
    let nms_done = nms_stream
        .record_event(None)
        .expect("Failed to record nms event");

    // Do not run nms until edge is complete.
    threshold_stream
        .wait(&nms_done)
        .expect("Failed to wait for nms event");

    let low_threshold: f32 = 0.022;
    let high_threshold: f32 = 0.045;

    // Launch the threshold kernel.
    unsafe {
        module.double_threshold(
            &threshold_stream,
            LaunchConfig::for_num_elems(N as u32),
            &thin_edges_dev,
            &mut edge_classes_dev,
            low_threshold,
            high_threshold,
        )
    }
    .expect("Threshold kernal launch failed");

    // Record when the threshold kernal has completed its work.
    let threshold_done = threshold_stream
        .record_event(None)
        .expect("Failed to threshold edge event");

    // Do not run nms until threshold is complete.
    hysteresis_stream
        .wait(&threshold_done)
        .expect("Failed to wait for threshold event");

    // Launch the hysteresis kernel.
    for _ in 0..64 {
        unsafe {
            module.hysteresis(
                &hysteresis_stream,
                LaunchConfig::for_num_elems(N as u32),
                &edge_classes_dev,
                &connected_edges_dev,
                &mut connected_edges_next_dev,
                w,
                h,
            )
        }
        .expect("Hysteresis kernel launch failed");

        std::mem::swap(&mut connected_edges_dev, &mut connected_edges_next_dev);
    }

    // Get the results
    let grayscale_host = grayscale_dev.to_host_vec(&grayscale_stream).unwrap();
    let edges_host = edges_dev.to_host_vec(&edge_stream).unwrap();
    let thin_edges_host = thin_edges_dev.to_host_vec(&nms_stream).unwrap();
    let edge_classes_host = edge_classes_dev.to_host_vec(&threshold_stream).unwrap();
    let connected_edges_host = connected_edges_dev.to_host_vec(&hysteresis_stream).unwrap();

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

    let grayscale_img = image::GrayImage::from_raw(rgba.width(), rgba.height(), grayscale)
        .expect("grayscale buffer has the wrong length");
    let edges_img = image::GrayImage::from_raw(rgba.width(), rgba.height(), edges)
        .expect("edges buffer has the wrong length");
    let thin_edges_img = image::GrayImage::from_raw(rgba.width(), rgba.height(), thin_edges)
        .expect("thin_edges buffer has the wrong length");
    let edge_classes_img = image::GrayImage::from_raw(rgba.width(), rgba.height(), edge_classes)
        .expect("edge_classes buffer has the wrong length");
    let connected_edges_img =
        image::GrayImage::from_raw(rgba.width(), rgba.height(), connected_edges)
            .expect("connected_edges buffer has the wrong length");

    grayscale_img.save("assets/circle-grayscale.png").unwrap();
    edges_img.save("assets/circle-edges.png").unwrap();
    thin_edges_img.save("assets/circle-thin-edges.png").unwrap();
    edge_classes_img
        .save("assets/circle-edge-classes.png")
        .unwrap();
    connected_edges_img
        .save("assets/circle-connected-edges.png")
        .unwrap();

    println!("Hello, world!");
}

fn normalized_f32_to_u8(values: &[f32], gain: f32) -> Vec<u8> {
    values
        .iter()
        .map(|&value| ((value * gain).clamp(0.0, 1.0) * 255.0) as u8)
        .collect()
}
