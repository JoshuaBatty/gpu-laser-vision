//! Conversion from a thinned edge image into vector and laser paths.

use std::collections::HashSet;

use image::{GrayImage, RgbImage};
use nannou::lyon::{math::point, path::Path};
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

/// A Lyon path and the equivalent line sequences accepted by `nannou_laser`.
pub struct LaserPath {
    lyon_path: Path,
    laser_points: Vec<LaserPoint>,
    laser_lines: Vec<Vec<LaserPoint>>,
}

impl LaserPath {
    /// The vector path used for on-screen rendering.
    pub fn lyon_path(&self) -> &Path {
        &self.lyon_path
    }

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

struct PixelLine {
    pixels: Vec<usize>,
    closed: bool,
}

/// Traces the non-zero pixels in a hysteresis image into ordered vector paths.
pub fn from_hysteresis(image: &GrayImage, edge_colors: &RgbImage) -> LaserPath {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let active: Vec<_> = image.as_raw().iter().map(|&value| value != 0).collect();
    let mut visited_edges = HashSet::new();
    let mut lines = Vec::new();
    let isolated_pixels: Vec<_> = (0..active.len())
        .filter(|&pixel| {
            active[pixel] && neighbors(pixel, width, height, &active).is_empty()
        })
        .collect();

    // Trace paths that begin or end at endpoints and junctions first.
    for start in 0..active.len() {
        if !active[start] || neighbors(start, width, height, &active).len() == 2 {
            continue;
        }
        for next in neighbors(start, width, height, &active) {
            if !visited_edges.contains(&edge(start, next)) {
                lines.push(trace_line(
                    start,
                    next,
                    width,
                    height,
                    &active,
                    &mut visited_edges,
                ));
            }
        }
    }

    // Any edges left belong to closed loops where every pixel has degree two.
    for start in 0..active.len() {
        if !active[start] {
            continue;
        }
        for next in neighbors(start, width, height, &active) {
            if !visited_edges.contains(&edge(start, next)) {
                lines.push(trace_line(
                    start,
                    next,
                    width,
                    height,
                    &active,
                    &mut visited_edges,
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

    let mut lyon_builder = Path::builder();
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

        lyon_builder.begin(point(
            laser_line[0].position[0],
            laser_line[0].position[1],
        ));
        for laser_point in &laser_line[1..] {
            lyon_builder.line_to(point(laser_point.position[0], laser_point.position[1]));
        }
        lyon_builder.end(line.closed);

        if line.closed {
            laser_line.push(laser_line[0]);
        }
        laser_lines.push(laser_line);
    }

    LaserPath {
        lyon_path: lyon_builder.build(),
        laser_points,
        laser_lines,
    }
}

fn trace_line(
    start: usize,
    next: usize,
    width: usize,
    height: usize,
    active: &[bool],
    visited_edges: &mut HashSet<(usize, usize)>,
) -> PixelLine {
    let mut pixels = vec![start];
    let mut previous = start;
    let mut current = next;
    visited_edges.insert(edge(start, next));

    while current != start {
        pixels.push(current);
        let adjacent = neighbors(current, width, height, active);
        if adjacent.len() != 2 {
            break;
        }
        let Some(next) = adjacent
            .into_iter()
            .find(|&candidate| candidate != previous && !visited_edges.contains(&edge(current, candidate)))
        else {
            break;
        };
        visited_edges.insert(edge(current, next));
        previous = current;
        current = next;
    }

    PixelLine {
        pixels,
        closed: current == start,
    }
}

fn neighbors(pixel: usize, width: usize, height: usize, active: &[bool]) -> Vec<usize> {
    let x = (pixel % width) as isize;
    let y = (pixel / width) as isize;
    let mut result = Vec::with_capacity(8);

    for (dx, dy) in NEIGHBOR_OFFSETS {
        let next_x = x + dx;
        let next_y = y + dy;
        if next_x < 0 || next_y < 0 || next_x >= width as isize || next_y >= height as isize {
            continue;
        }
        let next = next_y as usize * width + next_x as usize;
        if !active[next] {
            continue;
        }

        // Prefer orthogonal links when present to avoid redundant diagonal triangles.
        if dx != 0 && dy != 0 {
            let horizontal = y as usize * width + next_x as usize;
            let vertical = next_y as usize * width + x as usize;
            if active[horizontal] || active[vertical] {
                continue;
            }
        }
        result.push(next);
    }

    result
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

fn edge(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}
