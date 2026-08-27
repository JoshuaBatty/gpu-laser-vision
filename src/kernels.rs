//! CUDA kernels for extracting colourized laser edges.

use cuda_device::{DisjointSlice, cuda_module, kernel};
pub use device::*;

#[cuda_module]
mod device {
    use super::*;

    const COLOR_SAMPLE_DISTANCE: usize = 2;

    /// Converts packed row-major RGBA pixels to normalized luminance.
    #[kernel]
    pub fn convert_to_grayscale(rgba: &[u8], mut b: DisjointSlice<f32>) {
        if let Some((b_elem, idx)) = b.get_mut_indexed() {
            let source = idx.get() * 4;
            let red = rgba[source] as f32 / 255.0;
            let green = rgba[source + 1] as f32 / 255.0;
            let blue = rgba[source + 2] as f32 / 255.0;
            *b_elem = 0.299 * red + 0.587 * green + 0.114 * blue;
        }
    }

    /// Writes normalized Scharr magnitude and its horizontal and vertical gradients.
    #[kernel]
    pub fn scharr(
        gray: &[f32],
        mut b: DisjointSlice<f32>,
        mut grad_x: DisjointSlice<f32>,
        mut grad_y: DisjointSlice<f32>,
        w: usize,
        h: usize,
    ) {
        if let (Some((edge, idx)), Some((grad_x_out, _)), Some((grad_y_out, _))) = (
            b.get_mut_indexed(),
            grad_x.get_mut_indexed(),
            grad_y.get_mut_indexed(),
        ) {
            let i = idx.get();
            let x = i % w;
            let y = i / w;

            // Give border pixels no edge value.
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                *edge = 0.0;
                *grad_x_out = 0.0;
                *grad_y_out = 0.0;
                return;
            }

            // Read the eight neighbouring grayscale pixels.
            let top_left = gray[i - w - 1];
            let top = gray[i - w];
            let top_right = gray[i - w + 1];
            let left = gray[i - 1];
            let right = gray[i + 1];
            let bottom_left = gray[i + w - 1];
            let bottom = gray[i + w];
            let bottom_right = gray[i + w + 1];

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
        w: usize,
        thresholds: &[f32],
    ) {
        if let Some((edge_color, idx)) = edge_colors.get_mut_indexed() {
            let i = idx.get();
            let gx = grad_x[i];
            let gy = grad_y[i];
            let strength = edges[i];
            if strength < thresholds[0] || strength > thresholds[1] {
                *edge_color = 0;
                return;
            }

            let x = i % w;
            let y = i / w;
            let h = rgba.len() / 4 / w;

            // Quantize the gradient normal to horizontal, vertical, or diagonal.
            let (dx, dy): (isize, isize) = if gx.abs() >= gy.abs() {
                if gy.abs() * 2.0 <= gx.abs() {
                    (1, 0)
                } else if gx * gy >= 0.0 {
                    (1, 1)
                } else {
                    (1, -1)
                }
            } else if gx.abs() * 2.0 <= gy.abs() {
                (0, 1)
            } else if gx * gy >= 0.0 {
                (1, 1)
            } else {
                (1, -1)
            };

            let plus_x = offset_coordinate(x, dx, COLOR_SAMPLE_DISTANCE, w);
            let plus_y = offset_coordinate(y, dy, COLOR_SAMPLE_DISTANCE, h);
            let minus_x = offset_coordinate(x, -dx, COLOR_SAMPLE_DISTANCE, w);
            let minus_y = offset_coordinate(y, -dy, COLOR_SAMPLE_DISTANCE, h);

            let center = i;
            let plus = plus_y * w + plus_x;
            let minus = minus_y * w + minus_x;
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
            *edge_color = rgba[source] as u32
                | (rgba[source + 1] as u32) << 8
                | (rgba[source + 2] as u32) << 16
                | 0xff << 24;
        }
    }

    fn offset_coordinate(
        coordinate: usize,
        direction: isize,
        distance: usize,
        limit: usize,
    ) -> usize {
        if direction < 0 {
            coordinate.saturating_sub(distance)
        } else if direction > 0 {
            (coordinate + distance).min(limit - 1)
        } else {
            coordinate
        }
    }

    fn color_score(rgba: &[u8], pixel: usize) -> u16 {
        let source = pixel * 4;
        let red = rgba[source];
        let green = rgba[source + 1];
        let blue = rgba[source + 2];
        let brightest = red.max(green).max(blue);
        let darkest = red.min(green).min(blue);
        brightest as u16 * 2 + (brightest - darkest) as u16
    }
}
