//! Conversion from a thinned edge image into laser paths.

use image::{GrayImage, RgbImage};
use nannou_laser::Point as LaserPoint;

const NEIGHBOR_OFFSETS: [(isize, isize); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// Point and line sequences accepted by `nannou_laser`.
pub struct LaserPath {
    laser_points: Vec<LaserPoint>,
    laser_lines: Vec<Vec<LaserPoint>>,
}

impl LaserPath {
    /// Isolated points ready to submit with `nannou_laser::Frame::add_points`.
    pub fn laser_points(&self) -> &[LaserPoint] {
        &self.laser_points
    }

    /// Ordered line sequences ready to submit with `nannou_laser::Frame::add_lines`.
    pub fn laser_lines(&self) -> &[Vec<LaserPoint>] {
        &self.laser_lines
    }

    /// Total number of points that would be submitted to a laser frame.
    pub fn point_count(&self) -> usize {
        self.laser_points.len() + self.laser_lines.iter().map(Vec::len).sum::<usize>()
    }
}

/// Pixel indices forming one open path or closed loop.
struct PixelLine {
    pixels: Vec<usize>,
    closed: bool,
}

/// Adjacent active pixel and its index in the clockwise offset table.
#[derive(Clone, Copy)]
struct Neighbor {
    pixel: usize,
    direction: u8,
}

/// Fixed-capacity neighborhood of one mask pixel.
struct Neighbors {
    entries: [Neighbor; 8],
    len: usize,
}

impl Neighbors {
    fn iter(&self) -> impl Iterator<Item = Neighbor> + '_ {
        self.entries[..self.len].iter().copied()
    }
}

/// Traces the supplied non-zero mask pixels into ordered vector paths.
///
/// `active_pixels` must be sorted in row-major order and contain every non-zero
/// pixel in `image` exactly once.
pub fn from_edge_mask(
    image: &GrayImage,
    edge_colors: &RgbImage,
    active_pixels: &[usize],
) -> LaserPath {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let pixels = image.as_raw();
    debug_assert_eq!(image.dimensions(), edge_colors.dimensions());
    debug_assert!(active_pixels.windows(2).all(|pair| pair[0] < pair[1]));
    debug_assert!(
        active_pixels
            .iter()
            .all(|&pixel| pixels.get(pixel).is_some_and(|&value| value != 0))
    );
    let mut degrees = vec![0; pixels.len()];
    let mut isolated_pixels = Vec::new();
    for &pixel in active_pixels {
        let degree = neighbors(pixel, width, height, pixels).len;
        degrees[pixel] = degree as u8;
        if degree == 0 {
            isolated_pixels.push(pixel);
        }
    }

    // Each bit records whether the edge in the corresponding neighbor
    // direction has been visited. This is both denser and faster than hashing
    // a pair of pixel indices for every edge in a frame-sized image.
    let mut visited_directions = vec![0_u8; pixels.len()];
    let mut lines = Vec::new();

    // Trace paths that begin or end at endpoints and junctions first.
    for &start in active_pixels {
        if degrees[start] == 2 {
            continue;
        }
        for neighbor in neighbors(start, width, height, pixels).iter() {
            if !edge_was_visited(&visited_directions, start, neighbor.direction) {
                lines.push(trace_line(
                    start,
                    neighbor,
                    width,
                    height,
                    pixels,
                    &degrees,
                    &mut visited_directions,
                ));
            }
        }
    }

    // Any edges left belong to closed loops where every pixel has degree two.
    for &start in active_pixels {
        for neighbor in neighbors(start, width, height, pixels).iter() {
            if !edge_was_visited(&visited_directions, start, neighbor.direction) {
                lines.push(trace_line(
                    start,
                    neighbor,
                    width,
                    height,
                    pixels,
                    &degrees,
                    &mut visited_directions,
                ));
            }
        }
    }

    let lines: Vec<_> = lines
        .into_iter()
        .map(|line| PixelLine {
            pixels: collapse_straight_runs(line.pixels, line.closed, width),
            closed: line.closed,
        })
        .filter(|line| line.pixels.len() >= if line.closed { 3 } else { 2 })
        .collect();

    let laser_points = isolated_pixels
        .into_iter()
        .map(|pixel| {
            LaserPoint::new(
                normalize(pixel, width, height),
                laser_color(pixel, edge_colors),
            )
        })
        .collect();
    let mut laser_lines = Vec::with_capacity(lines.len());

    for line in lines {
        let mut laser_line: Vec<_> = line
            .pixels
            .into_iter()
            .map(|pixel| {
                LaserPoint::new(
                    normalize(pixel, width, height),
                    laser_color(pixel, edge_colors),
                )
            })
            .collect();

        if line.closed {
            laser_line.push(laser_line[0]);
        }
        laser_lines.push(laser_line);
    }

    LaserPath {
        laser_points,
        laser_lines,
    }
}

fn trace_line(
    start: usize,
    next: Neighbor,
    width: usize,
    height: usize,
    pixels: &[u8],
    degrees: &[u8],
    visited_directions: &mut [u8],
) -> PixelLine {
    let mut line_pixels = vec![start];
    let mut previous = start;
    let mut current = next.pixel;
    visit_edge(visited_directions, start, next);

    while current != start {
        line_pixels.push(current);
        if degrees[current] != 2 {
            break;
        }
        let Some(next) = neighbors(current, width, height, pixels)
            .iter()
            .find(|neighbor| {
                neighbor.pixel != previous
                    && !edge_was_visited(visited_directions, current, neighbor.direction)
            })
        else {
            break;
        };
        visit_edge(visited_directions, current, next);
        previous = current;
        current = next.pixel;
    }

    PixelLine {
        pixels: line_pixels,
        closed: current == start,
    }
}

fn neighbors(pixel: usize, width: usize, height: usize, pixels: &[u8]) -> Neighbors {
    let x = (pixel % width) as isize;
    let y = (pixel / width) as isize;
    let mut result = Neighbors {
        entries: [Neighbor {
            pixel: 0,
            direction: 0,
        }; 8],
        len: 0,
    };

    for (direction, (dx, dy)) in NEIGHBOR_OFFSETS.into_iter().enumerate() {
        let next_x = x + dx;
        let next_y = y + dy;
        if next_x < 0 || next_y < 0 || next_x >= width as isize || next_y >= height as isize {
            continue;
        }
        let next = next_y as usize * width + next_x as usize;
        if pixels[next] == 0 {
            continue;
        }

        // Prefer orthogonal links when present to avoid redundant diagonal triangles.
        if dx != 0 && dy != 0 {
            let horizontal = y as usize * width + next_x as usize;
            let vertical = next_y as usize * width + x as usize;
            if pixels[horizontal] != 0 || pixels[vertical] != 0 {
                continue;
            }
        }
        result.entries[result.len] = Neighbor {
            pixel: next,
            direction: direction as u8,
        };
        result.len += 1;
    }

    result
}

fn edge_was_visited(visited_directions: &[u8], pixel: usize, direction: u8) -> bool {
    visited_directions[pixel] & (1 << direction) != 0
}

fn visit_edge(visited_directions: &mut [u8], pixel: usize, neighbor: Neighbor) {
    visited_directions[pixel] |= 1 << neighbor.direction;
    // Opposite directions are four positions apart in NEIGHBOR_OFFSETS.
    visited_directions[neighbor.pixel] |= 1 << ((neighbor.direction + 4) % 8);
}

fn collapse_straight_runs(pixels: Vec<usize>, closed: bool, width: usize) -> Vec<usize> {
    if pixels.len() <= 2 {
        return pixels;
    }

    let mut simplified = Vec::with_capacity(pixels.len());
    if closed {
        for index in 0..pixels.len() {
            let previous = pixels[(index + pixels.len() - 1) % pixels.len()];
            let current = pixels[index];
            let next = pixels[(index + 1) % pixels.len()];
            if direction(previous, current, width) != direction(current, next, width) {
                simplified.push(current);
            }
        }
        if simplified.len() < 3 {
            return pixels;
        }
    } else {
        simplified.push(pixels[0]);
        for points in pixels.windows(3) {
            if direction(points[0], points[1], width) != direction(points[1], points[2], width) {
                simplified.push(points[1]);
            }
        }
        simplified.push(*pixels.last().unwrap());
    }

    simplified
}

fn direction(from: usize, to: usize, width: usize) -> (isize, isize) {
    let from = ((from % width) as isize, (from / width) as isize);
    let to = ((to % width) as isize, (to / width) as isize);
    (to.0 - from.0, to.1 - from.1)
}

fn normalize(pixel: usize, width: usize, height: usize) -> [f32; 2] {
    let x = (pixel % width) as f32 / width.saturating_sub(1).max(1) as f32;
    let y = (pixel / width) as f32 / height.saturating_sub(1).max(1) as f32;
    [x * 2.0 - 1.0, 1.0 - y * 2.0]
}

fn laser_color(pixel: usize, edge_colors: &RgbImage) -> [f32; 3] {
    let source = pixel * 3;
    let colors = edge_colors.as_raw();
    [
        colors[source] as f32 / 255.0,
        colors[source + 1] as f32 / 255.0,
        colors[source + 2] as f32 / 255.0,
    ]
}
