//! CUDA kernels for extracting colourized laser edges.

use cuda_device::{DisjointSlice, cuda_module, kernel};
pub use device::*;

// `#[kernel]` supplies the CUDA entry-point linkage expected by the code generator.
#[allow(clippy::no_mangle_with_rust_abi)]
#[cuda_module]
mod device {
    use core::cmp::Ordering;

    use super::{DisjointSlice, kernel};

    const COLOR_SAMPLE_DISTANCE: usize = 2;

    /// Converts packed row-major RGBA pixels to normalized luminance.
    #[kernel]
    pub fn convert_to_grayscale(rgba: &[u8], mut grayscale: DisjointSlice<f32>) {
        if let Some((output, idx)) = grayscale.get_mut_indexed() {
            let source = idx.get() * 4;
            let red = f32::from(rgba[source]) / 255.0;
            let green = f32::from(rgba[source + 1]) / 255.0;
            let blue = f32::from(rgba[source + 2]) / 255.0;
            *output = 0.299 * red + 0.587 * green + 0.114 * blue;
        }
    }

    /// Writes normalized Scharr magnitude and its horizontal and vertical gradients.
    #[kernel]
    pub fn scharr(
        gray: &[f32],
        mut magnitude: DisjointSlice<f32>,
        mut grad_x: DisjointSlice<f32>,
        mut grad_y: DisjointSlice<f32>,
        width: usize,
        height: usize,
    ) {
        if let (Some((edge, idx)), Some((grad_x_out, _)), Some((grad_y_out, _))) = (
            magnitude.get_mut_indexed(),
            grad_x.get_mut_indexed(),
            grad_y.get_mut_indexed(),
        ) {
            let index = idx.get();
            let x = index % width;
            let y = index / width;

            // Give border pixels no edge value.
            if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
                *edge = 0.0;
                *grad_x_out = 0.0;
                *grad_y_out = 0.0;
                return;
            }

            // Read the eight neighbouring grayscale pixels.
            let top_left = gray[index - width - 1];
            let top = gray[index - width];
            let top_right = gray[index - width + 1];
            let left = gray[index - 1];
            let right = gray[index + 1];
            let bottom_left = gray[index + width - 1];
            let bottom = gray[index + width];
            let bottom_right = gray[index + width + 1];

            // Scharr horizontal and vertical gradients.
            let gx = (-3.0 * top_left + 3.0 * top_right - 10.0 * left + 10.0 * right
                - 3.0 * bottom_left
                + 3.0 * bottom_right)
                / 4.0;

            let gy = (-3.0 * top_left - 10.0 * top - 3.0 * top_right
                + 3.0 * bottom_left
                + 10.0 * bottom
                + 3.0 * bottom_right)
                / 4.0;

            *grad_x_out = gx;
            *grad_y_out = gy;

            // Rotation-friendly L2 edge strength.
            let strength = (gx * gx + gy * gy).sqrt();
            *edge = strength.min(1.0);
        }
    }

    /// Converts a normalized floating-point stage into display-ready bytes on the GPU.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[kernel]
    pub fn normalized_f32_to_u8(input: &[f32], mut output: DisjointSlice<u8>) {
        if let Some((output, idx)) = output.get_mut_indexed() {
            *output = (input[idx.get()].clamp(0.0, 1.0) * 255.0) as u8;
        }
    }

    /// Thresholds Scharr magnitudes and recovers source colours in one pass.
    #[kernel]
    pub fn extract_laser_edges(
        rgba: &[u8],
        edges: &[f32],
        grad_x: &[f32],
        grad_y: &[f32],
        mut edge_colors: DisjointSlice<u32>,
        width: usize,
        thresholds: &[f32],
    ) {
        if let Some((edge_color, idx)) = edge_colors.get_mut_indexed() {
            let index = idx.get();
            let gradient_x = grad_x[index];
            let gradient_y = grad_y[index];
            let strength = edges[index];
            if strength < thresholds[0] || strength > thresholds[1] {
                *edge_color = 0;
                return;
            }

            let x = index % width;
            let y = index / width;
            let height = rgba.len() / 4 / width;

            // Quantize the gradient normal to horizontal, vertical, or diagonal.
            let (dx, dy): (isize, isize) = if gradient_x.abs() >= gradient_y.abs() {
                if gradient_y.abs() * 2.0 <= gradient_x.abs() {
                    (1, 0)
                } else if gradient_x * gradient_y >= 0.0 {
                    (1, 1)
                } else {
                    (1, -1)
                }
            } else if gradient_x.abs() * 2.0 <= gradient_y.abs() {
                (0, 1)
            } else if gradient_x * gradient_y >= 0.0 {
                (1, 1)
            } else {
                (1, -1)
            };

            let plus_x = offset_coordinate(x, dx, COLOR_SAMPLE_DISTANCE, width);
            let plus_y = offset_coordinate(y, dy, COLOR_SAMPLE_DISTANCE, height);
            let minus_x = offset_coordinate(x, -dx, COLOR_SAMPLE_DISTANCE, width);
            let minus_y = offset_coordinate(y, -dy, COLOR_SAMPLE_DISTANCE, height);

            let center = index;
            let plus = plus_y * width + plus_x;
            let minus = minus_y * width + minus_x;
            let mut selected = center;
            let mut selected_score = color_score(rgba, center);
            let plus_score = color_score(rgba, plus);
            if plus_score > selected_score {
                selected = plus;
                selected_score = plus_score;
            }
            if color_score(rgba, minus) > selected_score {
                selected = minus;
            }

            let source = selected * 4;
            *edge_color = u32::from(rgba[source])
                | u32::from(rgba[source + 1]) << 8
                | u32::from(rgba[source + 2]) << 16
                | 0xff << 24;
        }
    }

    fn offset_coordinate(
        coordinate: usize,
        direction: isize,
        distance: usize,
        limit: usize,
    ) -> usize {
        match direction.cmp(&0) {
            Ordering::Less => coordinate.saturating_sub(distance),
            Ordering::Equal => coordinate,
            Ordering::Greater => (coordinate + distance).min(limit - 1),
        }
    }

    fn color_score(rgba: &[u8], pixel: usize) -> u16 {
        let source = pixel * 4;
        let red = rgba[source];
        let green = rgba[source + 1];
        let blue = rgba[source + 2];
        let brightest = red.max(green).max(blue);
        let darkest = red.min(green).min(blue);
        u16::from(brightest) * 2 + u16::from(brightest - darkest)
    }
}
