//! CUDA kernels for the Canny edge detection pipeline.

use cuda_device::{DisjointSlice, cuda_module, kernel};
pub use device::*;

#[cuda_module]
mod device {
    use super::*;

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

    #[kernel]
    pub fn non_maximum_suppression(
        edges: &[f32],
        grad_x: &[f32],
        grad_y: &[f32],
        mut thin_edges: DisjointSlice<f32>,
        w: usize,
        h: usize,
    ) {
        if let Some((thin_edge, idx)) = thin_edges.get_mut_indexed() {
            let i = idx.get();
            let x = i % w;
            let y = i / w;

            // Borders have no complete neighbourhood.
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                *thin_edge = 0.0;
                return;
            }

            let gx = grad_x[i];
            let gy = grad_y[i];
            let current = edges[i];

            // Pick the two neighbours along the closest gradient direction.
            let (before, after) = if gx.abs() >= gy.abs() {
                if gy.abs() * 2.0 <= gx.abs() {
                    (i - 1, i + 1) // horizontal
                } else if gx * gy >= 0.0 {
                    (i - w - 1, i + w + 1) // northwest ↔ southeast
                } else {
                    (i - w + 1, i + w - 1) // northeast ↔ southwest
                }
            } else if gx.abs() * 2.0 <= gy.abs() {
                (i - w, i + w) // vertical
            } else if gx * gy >= 0.0 {
                (i - w - 1, i + w + 1) // northwest ↔ southeast
            } else {
                (i - w + 1, i + w - 1) // northeast ↔ southwest
            };

            // Keep only local maxima; suppress every other edge pixel.
            *thin_edge = if current >= edges[before] && current >= edges[after] {
                current
            } else {
                0.0
            };
        }
    }

    #[kernel]
    pub fn double_threshold(
        thin_edges: &[f32],
        mut edge_classes: DisjointSlice<f32>,
        low_threshold: f32,
        high_threshold: f32,
    ) {
        if let Some((edge_class, idx)) = edge_classes.get_mut_indexed() {
            let strength = thin_edges[idx.get()];

            // 0.0 = no edge, 0.5 = weak edge, 1.0 = strong edge.
            *edge_class = if strength >= high_threshold {
                1.0
            } else if strength >= low_threshold {
                0.5
            } else {
                0.0
            };
        }
    }

    #[kernel]
    pub fn hysteresis(
        edge_classes: &[f32],
        connected_edges: &[f32],
        mut connected_edges_next: DisjointSlice<f32>,
        w: usize,
        h: usize,
    ) {
        if let Some((output, idx)) = connected_edges_next.get_mut_indexed() {
            let i = idx.get();
            let x = i % w;
            let y = i / w;

            // Strong edges and already-connected edges stay connected.
            if edge_classes[i] == 1.0 || connected_edges[i] == 1.0 {
                *output = 1.0;
                return;
            }

            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                *output = 0.0;
                return;
            }

            // Promote weak pixels that touch the previous connected result.
            let touches_connected = connected_edges[i - w - 1] == 1.0
                || connected_edges[i - w] == 1.0
                || connected_edges[i - w + 1] == 1.0
                || connected_edges[i - 1] == 1.0
                || connected_edges[i + 1] == 1.0
                || connected_edges[i + w - 1] == 1.0
                || connected_edges[i + w] == 1.0
                || connected_edges[i + w + 1] == 1.0;

            *output = if edge_classes[i] == 0.5 && touches_connected {
                1.0
            } else {
                0.0
            };
        }
    }

    /// Converts a normalized floating-point stage into display-ready bytes on the GPU.
    #[kernel]
    pub fn normalized_f32_to_u8(input: &[f32], mut output: DisjointSlice<u8>, gain: f32) {
        if let Some((output, idx)) = output.get_mut_indexed() {
            *output = ((input[idx.get()] * gain).clamp(0.0, 1.0) * 255.0) as u8;
        }
    }

    // Recover original image colours on the GPU before the live pipeline leaves image space.
    #[kernel]
    pub fn colorize_edges(
        rgba: &[u8],
        connected_edges: &[f32],
        grad_x: &[f32],
        grad_y: &[f32],
        mut edge_colors: DisjointSlice<u32>,
        w: usize,
        h: usize,
        sample_distance: usize,
    ) {
        if let Some((edge_color, idx)) = edge_colors.get_mut_indexed() {
            let i = idx.get();
            if connected_edges[i] == 0.0 {
                *edge_color = 0;
                return;
            }

            let x = i % w;
            let y = i / w;
            let gx = grad_x[i];
            let gy = grad_y[i];

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

            let plus_x = offset_coordinate(x, dx, sample_distance, w);
            let plus_y = offset_coordinate(y, dy, sample_distance, h);
            let minus_x = offset_coordinate(x, -dx, sample_distance, w);
            let minus_y = offset_coordinate(y, -dy, sample_distance, h);

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
                | (rgba[source + 2] as u32) << 16;
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
