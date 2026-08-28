//! Ether Dream discovery, scanner-frame compilation, and live output.
//!
//! Image-derived paths are reduced to bounded physical frames on the app
//! thread, then a dedicated worker continuously feeds the DAC's TCP buffer.

use std::{
    io::ErrorKind,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use nannou_laser::{
    DacId, DetectedDac, Point, RawPoint,
    ether_dream::{
        self,
        protocol::{DacBroadcast, DacPoint, DacStatus},
    },
};

use crate::path_generation::LaserPath;

const DISCOVERY_TIMEOUT: Duration = Duration::from_millis(500);
const DIRECT_CONNECT_DELAY: Duration = Duration::from_secs(1);
const DIRECT_CONNECT_RETRY: Duration = Duration::from_secs(2);
const TCP_TIMEOUT: Duration = Duration::from_secs(3);
const DEFAULT_ETHER_DREAM_IP: IpAddr = IpAddr::V4(Ipv4Addr::new(192, 168, 0, 2));
const ETHER_DREAM_BUFFER_CAPACITY: u16 = 1_800;
const ETHER_DREAM_MAX_POINT_RATE: u32 = 100_000;
const LASER_POINT_RATE: u32 = 30_000;
const LASER_FRAME_RATE: u32 = 30;
const LASER_TARGET_POINTS: u32 = LASER_POINT_RATE / LASER_FRAME_RATE;
const MAX_CONTOUR_LINES: usize = 12;
const MAX_CONTOUR_POINTS: usize = 240;
const MAX_EDGE_CANDIDATES: usize = 48;
const MAX_EDGE_LINES: usize = 6;
const MAX_EDGE_CONTROL_POINTS: usize = 160;
const EDGE_JOIN_DISTANCE: f32 = 0.008;

/// Packing policy for geometry sent to the physical scanner.
#[derive(Clone, Copy)]
pub enum FrameProfile {
    /// Fragmented edge geometry requiring aggressive path reduction.
    DenseEdges,
    /// A small set of already-coherent contour paths.
    Contour,
}

/// Current Ether Dream discovery or connection state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConnectionState {
    /// Waiting for discovery or the configured direct-IP probe.
    #[default]
    Searching,
    /// Opening the DAC frame stream.
    Connecting,
    /// The DAC frame stream is active.
    Streaming,
    /// Discovery ended without a DAC.
    Stopped,
    /// Discovery or connection failed.
    Error,
}

impl ConnectionState {
    /// Compact label suitable for the dashboard.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Searching => "SEARCHING",
            Self::Connecting => "CONNECTING",
            Self::Streaming => "CONNECTED",
            Self::Stopped => "STOPPED",
            Self::Error => "ERROR",
        }
    }
}

/// Display-ready snapshot of the Ether Dream connection.
#[derive(Clone, Debug, Default)]
pub struct EtherDreamStatus {
    state: ConnectionState,
    device: Option<String>,
    detail: Option<String>,
}

impl EtherDreamStatus {
    /// Current connection phase.
    pub const fn state(&self) -> ConnectionState {
        self.state
    }

    /// Detected Ether Dream MAC address or direct endpoint.
    pub fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }

    /// Connection error detail, when present.
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

/// Cloneable control handle for the frame consumed by the DAC thread.
///
/// Frame compilation is synchronous, while publication and enable state are
/// shared with the worker through short-lived synchronization primitives.
#[derive(Clone, Default)]
pub struct LaserControl {
    enabled: Arc<AtomicBool>,
    frame: Arc<Mutex<Arc<LaserFrame>>>,
}

impl LaserControl {
    /// Enables the installed path or requests blank output.
    ///
    /// Already-buffered DAC points drain before a disable becomes visible.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Compiles a path and publishes it for the next complete scanner frame.
    pub fn set_path(&self, path: &LaserPath, profile: FrameProfile) {
        let frame = Arc::new(LaserFrame {
            points: compile_laser_frame(path, profile),
        });
        *lock(&self.frame) = frame;
    }
}

/// Background Ether Dream connection and its live output control.
///
/// Discovery falls back to `ETHER_DREAM_IP` (default `192.168.0.2`) when LAN
/// broadcasts are unavailable. Dropping the stream requests shutdown and waits
/// for the worker, including any active discovery or TCP timeout.
pub struct EtherDreamStream {
    control: LaserControl,
    status: Arc<Mutex<EtherDreamStatus>>,
    stop_tx: mpsc::Sender<()>,
    worker: Option<thread::JoinHandle<()>>,
}

impl EtherDreamStream {
    /// Starts Ether Dream discovery with laser output disabled.
    pub fn start() -> Self {
        let control = LaserControl::default();
        let status = Arc::new(Mutex::new(EtherDreamStatus::default()));
        let worker_control = control.clone();
        let worker_status = Arc::clone(&status);
        let (stop_tx, stop_rx) = mpsc::channel();
        let worker = thread::spawn(move || run(worker_control, worker_status, stop_rx));

        Self {
            control,
            status,
            stop_tx,
            worker: Some(worker),
        }
    }

    /// Returns the handle used to enable and update output.
    pub const fn control(&self) -> &LaserControl {
        &self.control
    }

    /// Returns the latest discovery or connection status.
    pub fn status(&self) -> EtherDreamStatus {
        lock(&self.status).clone()
    }
}

impl Drop for EtherDreamStream {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        // Joining keeps network resources from outliving the owning app model.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Immutable wire-ready frame shared with the streaming worker.
#[derive(Default)]
struct LaserFrame {
    points: Vec<DacPoint>,
}

/// Discovers the DAC and owns each connection until shutdown, stop, or error.
#[allow(clippy::needless_pass_by_value)]
fn run(control: LaserControl, status: Arc<Mutex<EtherDreamStatus>>, stop_rx: mpsc::Receiver<()>) {
    let api = nannou_laser::Api::new();
    let mut detected_dacs = match api.detect_dacs() {
        Ok(detected_dacs) => detected_dacs,
        Err(error) => return set_error(&status, error),
    };
    if let Err(error) = detected_dacs.set_timeout(Some(DISCOVERY_TIMEOUT)) {
        return set_error(&status, error);
    }

    // Prefer discovery, then probe the known endpoint when WSL drops LAN broadcasts.
    let direct_ip = configured_direct_ip();
    let mut next_direct_attempt = Instant::now() + DIRECT_CONNECT_DELAY;
    let mut reported_direct_failure = false;
    let (dac, device) = loop {
        if stop_rx.try_recv().is_ok() {
            return;
        }
        match detected_dacs.next() {
            Some(Ok(dac)) => {
                let device = dac_address(&dac);
                break (dac, device);
            }
            Some(Err(error))
                if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Some(Err(error)) => return set_error(&status, error),
            None => {
                set_state(&status, ConnectionState::Stopped);
                return;
            }
        }

        if Instant::now() >= next_direct_attempt {
            match connect_direct(direct_ip) {
                Ok(dac) => break (dac, direct_ip.to_string()),
                Err(error) => {
                    if !reported_direct_failure {
                        eprintln!("Ether Dream direct probe at {direct_ip} failed: {error}");
                        reported_direct_failure = true;
                    }
                    next_direct_attempt = Instant::now() + DIRECT_CONNECT_RETRY;
                }
            }
        }
    };

    set_device(&status, device);
    set_state(&status, ConnectionState::Connecting);
    if let Err(error) = stream_frames(&dac, &control, &status, &stop_rx) {
        set_error(&status, error);
    }
}

/// Keeps the DAC near its latency target and changes geometry at frame boundaries.
fn stream_frames(
    dac: &DetectedDac,
    control: &LaserControl,
    status: &Mutex<EtherDreamStatus>,
    stop_rx: &mpsc::Receiver<()>,
) -> Result<(), ether_dream::dac::stream::CommunicationError> {
    let DetectedDac::EtherDream {
        broadcast,
        source_addr,
    } = dac;
    let mut stream =
        ether_dream::dac::stream::connect_timeout(broadcast, source_addr.ip(), TCP_TIMEOUT)?;
    stream.set_timeout(Some(TCP_TIMEOUT))?;
    stream.queue_commands().prepare_stream().submit()?;

    let buffer_capacity = stream.dac().buffer_capacity;
    let point_rate = LASER_POINT_RATE.min(stream.dac().max_point_rate);
    // Maintain roughly 16 ms of queued data: enough to avoid underruns without
    // making controls feel delayed.
    let latency_points = point_rate / 60;
    let initial_points = remaining_buffer_capacity(&stream);
    // Ether Dream requires preparation before data. A full blank prefill gives
    // playback safe points to consume from the instant `begin` takes effect.
    stream
        .queue_commands()
        .data((0..initial_points).map(|_| centered_blank()))
        .begin(0, point_rate)
        .submit()?;
    set_state(status, ConnectionState::Streaming);

    let blank_frame = Arc::new(LaserFrame {
        points: vec![centered_blank()],
    });
    let mut current_frame = latest_frame(control, &blank_frame);
    let mut frame_index = 0;
    let mut wire_points = Vec::with_capacity(latency_points as usize);

    loop {
        match stop_rx.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {}
        }

        if !control.enabled.load(Ordering::Acquire) {
            current_frame = Arc::clone(&blank_frame);
            frame_index = 0;
        }

        let fullness = stream.dac().status.buffer_fullness;
        let available = buffer_capacity.saturating_sub(1).saturating_sub(fullness);
        let points_needed = latency_points.saturating_sub(u32::from(fullness));
        let points_needed = points_needed.min(u32::from(available));

        wire_points.clear();
        for _ in 0..points_needed {
            wire_points.push(current_frame.points[frame_index]);
            frame_index += 1;
            if frame_index == current_frame.points.len() {
                frame_index = 0;
                // Never splice new geometry into the middle of an active frame.
                current_frame = latest_frame(control, &blank_frame);
            }
        }

        // Empty submissions still refresh DAC status and pace the worker.
        stream
            .queue_commands()
            .data(wire_points.drain(..))
            .submit()?;
    }

    let _ = stream.queue_commands().stop().submit();
    Ok(())
}

/// Selects the newest complete frame, falling back to a safe blank frame.
fn latest_frame(control: &LaserControl, blank_frame: &Arc<LaserFrame>) -> Arc<LaserFrame> {
    if !control.enabled.load(Ordering::Acquire) {
        return Arc::clone(blank_frame);
    }
    let frame = Arc::clone(&lock(&control.frame));
    if frame.points.is_empty() {
        Arc::clone(blank_frame)
    } else {
        frame
    }
}

/// Returns writable DAC slots while preserving the protocol's sentinel slot.
fn remaining_buffer_capacity(stream: &ether_dream::dac::stream::Stream) -> u16 {
    let dac = stream.dac();
    dac.buffer_capacity
        .saturating_sub(1)
        .saturating_sub(dac.status.buffer_fullness)
}

/// Creates the canonical blank wire point used while output is disabled.
fn centered_blank() -> DacPoint {
    raw_point_to_dac(RawPoint::centered_blank())
}

/// Applies the source-specific packing policy and produces wire-ready points.
fn compile_laser_frame(path: &LaserPath, profile: FrameProfile) -> Vec<DacPoint> {
    match profile {
        FrameProfile::DenseEdges => compile_dense_edge_frame(path),
        FrameProfile::Contour => compile_contour_frame(path),
    }
}

/// Reorders a small coherent contour before interpolating its scanner frame.
fn compile_contour_frame(path: &LaserPath) -> Vec<DacPoint> {
    let points = bounded_contour_points(path);
    let segments: Vec<_> = lasy::points_to_segments(points.iter().copied()).collect();
    if segments.is_empty() {
        return repeated_point(points.first().copied());
    }

    let point_graph = lasy::segments_to_point_graph(&points, segments);
    let euler_graph = lasy::point_graph_to_euler_graph(&point_graph);
    let circuit = lasy::euler_graph_to_euler_circuit(&points, &euler_graph);
    let raw_points: Vec<RawPoint> = lasy::interpolate_euler_circuit(
        &points,
        &circuit,
        &euler_graph,
        LASER_TARGET_POINTS,
        &lasy::InterpolationConfig::default(),
    );

    if raw_points.is_empty() {
        repeated_point(points.first().copied())
    } else {
        raw_points.into_iter().map(raw_point_to_dac).collect()
    }
}

/// Preserves the deterministic dense-edge order and adds conservative dwell.
fn compile_dense_edge_frame(path: &LaserPath) -> Vec<DacPoint> {
    let points = packed_edge_points(path);
    let segments: Vec<_> = lasy::points_to_segments(points.iter().copied()).collect();
    if segments.is_empty() {
        return repeated_point(points.first().copied());
    }

    let mut raw_points = Vec::new();
    let interpolation = lasy::InterpolationConfig {
        distance_per_point: lasy::InterpolationConfig::DEFAULT_DISTANCE_PER_POINT,
        blank_delay_points: 16,
        radians_per_point: 0.35,
    };
    lasy::interpolate_path(
        &points,
        segments,
        LASER_TARGET_POINTS,
        &interpolation,
        &mut raw_points,
    );

    if raw_points.is_empty() {
        repeated_point(points.first().copied())
    } else {
        raw_points.into_iter().map(raw_point_to_dac).collect()
    }
}

/// Limits coherent contour geometry before `lasy` expands it to the DAC budget.
fn bounded_contour_points(path: &LaserPath) -> Vec<Point> {
    let mut lines: Vec<_> = path
        .laser_lines()
        .iter()
        .filter(|line| line.len() >= 2)
        .map(|line| (line.as_slice(), line_length(line)))
        .collect();
    lines.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    lines.truncate(MAX_CONTOUR_LINES);

    if lines.is_empty() {
        return bounded_isolated_points(path.laser_points());
    }

    let minimum_points = lines.len() * 2;
    let distributable = MAX_CONTOUR_POINTS.saturating_sub(minimum_points);
    let total_length: f32 = lines.iter().map(|(_, length)| length).sum();
    let mut points = Vec::with_capacity(MAX_CONTOUR_POINTS + lines.len() * 2);

    for (line, length) in lines {
        let proportional = proportional_point_count(distributable, length, total_length);
        append_line(&mut points, line, (2 + proportional).min(line.len()));
    }

    points
}

/// Converts fragmented CUDA edges into a small, anchored set of ordered paths.
fn packed_edge_points(path: &LaserPath) -> Vec<Point> {
    let mut candidates: Vec<_> = path
        .laser_lines()
        .iter()
        .filter(|line| line.len() >= 2)
        .map(|line| (line.as_slice(), line_length(line)))
        .collect();
    candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    candidates.truncate(MAX_EDGE_CANDIDATES);
    let mut lines: Vec<_> = candidates
        .into_iter()
        .map(|(line, _)| line.to_vec())
        .collect();
    merge_edge_fragments(&mut lines);
    let mut lines: Vec<_> = lines
        .into_iter()
        .map(|line| {
            let length = line_length(&line);
            (line, length)
        })
        .collect();
    lines.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    lines.truncate(MAX_EDGE_LINES);

    if lines.is_empty() {
        return anchored_isolated_points(path.laser_points());
    }

    let lines = order_lines_by_travel(lines.into_iter().map(|(line, _)| line).collect());
    let lengths: Vec<f32> = lines.iter().map(|line| line_length(line)).collect();
    let total_length = lengths.iter().sum();
    let distributable = MAX_EDGE_CONTROL_POINTS.saturating_sub(lines.len() * 2);
    let mut points = Vec::with_capacity(MAX_EDGE_CONTROL_POINTS + lines.len() * 2 + 2);
    points.push(Point::centered_blank());

    for (line, length) in lines.iter().zip(lengths) {
        let sample_count = 2 + proportional_point_count(distributable, length, total_length);
        append_anchored_line(&mut points, line, sample_count.min(line.len()));
    }

    points.push(Point::centered_blank());
    points
}

/// Wraps isolated points in a stable blank center-to-center frame boundary.
fn anchored_isolated_points(isolated: &[Point]) -> Vec<Point> {
    let isolated = bounded_isolated_points(isolated);
    if isolated.is_empty() {
        return isolated;
    }

    let mut points = Vec::with_capacity(isolated.len() + 2);
    points.push(Point::centered_blank());
    points.extend(isolated);
    points.push(Point::centered_blank());
    points
}

/// Joins nearby endpoints to reduce blanked jumps between CUDA fragments.
fn merge_edge_fragments(lines: &mut Vec<Vec<Point>>) {
    loop {
        let mut best = None;
        for left in 0..lines.len() {
            for right in left + 1..lines.len() {
                for left_starts_at_join in [false, true] {
                    let left_point = endpoint(&lines[left], left_starts_at_join);
                    for right_starts_at_join in [true, false] {
                        let right_point = endpoint(&lines[right], right_starts_at_join);
                        let distance = point_distance(left_point, right_point);
                        if distance <= EDGE_JOIN_DISTANCE
                            && best
                                .is_none_or(|(_, _, _, _, best_distance)| distance < best_distance)
                        {
                            best = Some((
                                left,
                                right,
                                left_starts_at_join,
                                right_starts_at_join,
                                distance,
                            ));
                        }
                    }
                }
            }
        }

        let Some((left, right, reverse_left, right_starts_at_join, _)) = best else {
            break;
        };
        let mut right_line = lines.remove(right);
        let mut left_line = lines.remove(left);
        if reverse_left {
            left_line.reverse();
        }
        if !right_starts_at_join {
            right_line.reverse();
        }
        if left_line.last().map(|point| point.position)
            == right_line.first().map(|point| point.position)
        {
            right_line.remove(0);
        }
        left_line.extend(right_line);
        lines.push(left_line);
    }
}

/// Greedily orders and reverses paths to minimize blanked scanner travel.
fn order_lines_by_travel(mut lines: Vec<Vec<Point>>) -> Vec<Vec<Point>> {
    let mut ordered = Vec::with_capacity(lines.len());
    let mut current = [0.0, 0.0];

    while !lines.is_empty() {
        let mut best = (0, false, f32::INFINITY);
        for (index, line) in lines.iter().enumerate() {
            let to_start = position_distance(current, line[0].position);
            if to_start < best.2 {
                best = (index, false, to_start);
            }
            let to_end = position_distance(current, line[line.len() - 1].position);
            if to_end < best.2 {
                best = (index, true, to_end);
            }
        }

        let (index, reverse, _) = best;
        let mut line = lines.remove(index);
        if reverse {
            line.reverse();
        }
        current = line[line.len() - 1].position;
        ordered.push(line);
    }

    ordered
}

/// Adds one lit path with explicit blank points at both endpoints.
fn append_anchored_line(points: &mut Vec<Point>, line: &[Point], sample_count: usize) {
    if line.is_empty() || sample_count == 0 {
        return;
    }
    points.push(line[0].blanked());
    points.extend(resample_line(line, sample_count));
    points.push(line[line.len() - 1].blanked());
}

/// Resamples a path uniformly by distance rather than source vertex index.
#[allow(clippy::cast_precision_loss)]
fn resample_line(line: &[Point], sample_count: usize) -> Vec<Point> {
    if sample_count >= line.len() {
        return line.to_vec();
    }
    if sample_count <= 1 {
        return vec![line[0]];
    }

    let segment_lengths: Vec<_> = line
        .windows(2)
        .map(|pair| point_distance(pair[0], pair[1]))
        .collect();
    let total_length: f32 = segment_lengths.iter().sum();
    if total_length <= f32::EPSILON {
        return vec![line[0]; sample_count];
    }

    let mut sampled = Vec::with_capacity(sample_count);
    let mut segment = 0;
    let mut distance_before_segment = 0.0;
    for sample in 0..sample_count {
        let target = total_length * sample as f32 / (sample_count - 1) as f32;
        while segment + 1 < segment_lengths.len()
            && distance_before_segment + segment_lengths[segment] < target
        {
            distance_before_segment += segment_lengths[segment];
            segment += 1;
        }
        let segment_length = segment_lengths[segment];
        let amount = if segment_length <= f32::EPSILON {
            0.0
        } else {
            (target - distance_before_segment) / segment_length
        };
        sampled.push(interpolate_point(line[segment], line[segment + 1], amount));
    }
    sampled
}

/// Interpolates position and colour along a source path segment.
fn interpolate_point(start: Point, end: Point, amount: f32) -> Point {
    Point::new(
        std::array::from_fn(|axis| {
            (end.position[axis] - start.position[axis]).mul_add(amount, start.position[axis])
        }),
        std::array::from_fn(|channel| {
            (end.color[channel] - start.color[channel]).mul_add(amount, start.color[channel])
        }),
    )
}

fn endpoint(line: &[Point], start: bool) -> Point {
    if start { line[0] } else { line[line.len() - 1] }
}

fn point_distance(left: Point, right: Point) -> f32 {
    position_distance(left.position, right.position)
}

fn position_distance(left: [f32; 2], right: [f32; 2]) -> f32 {
    (right[0] - left[0]).hypot(right[1] - left[1])
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn proportional_point_count(available: usize, length: f32, total_length: f32) -> usize {
    if total_length <= f32::EPSILON {
        0
    } else {
        (available as f32 * length / total_length).floor() as usize
    }
}

/// Retains a bounded, evenly distributed subset of isolated points.
fn bounded_isolated_points(isolated: &[Point]) -> Vec<Point> {
    let point_count = isolated.len().min(MAX_CONTOUR_LINES);
    let mut points = Vec::with_capacity(point_count * 4);
    for sample in 0..point_count {
        let index = evenly_spaced_index(sample, point_count, isolated.len());
        append_line(&mut points, &isolated[index..=index], 1);
    }
    points
}

/// Appends one contour path with a blanked transition from the previous path.
fn append_line(points: &mut Vec<Point>, line: &[Point], sample_count: usize) {
    if line.is_empty() || sample_count == 0 {
        return;
    }

    let first = line[0];
    if let Some(previous) = points.last().copied() {
        points.push(previous.blanked());
        points.push(first.blanked());
    }

    if sample_count == 1 {
        // A duplicate lit point becomes a visible zero-length segment in `lasy`.
        points.extend([first, first]);
        return;
    }

    for sample in 0..sample_count {
        points.push(line[evenly_spaced_index(sample, sample_count, line.len())]);
    }
}

fn evenly_spaced_index(sample: usize, sample_count: usize, source_len: usize) -> usize {
    debug_assert!(sample < sample_count);
    debug_assert!(source_len > 0);
    if sample_count <= 1 {
        return 0;
    }
    sample * (source_len - 1) / (sample_count - 1)
}

fn line_length(line: &[Point]) -> f32 {
    line.windows(2)
        .map(|pair| {
            let dx = pair[1].position[0] - pair[0].position[0];
            let dy = pair[1].position[1] - pair[0].position[1];
            dx.hypot(dy)
        })
        .sum()
}

/// Produces a complete stationary frame for degenerate or empty geometry.
fn repeated_point(point: Option<Point>) -> Vec<DacPoint> {
    let point = point.map_or_else(RawPoint::centered_blank, |point| point.to_raw());
    let point = raw_point_to_dac(point);
    vec![point; LASER_TARGET_POINTS as usize]
}

/// Converts normalized nannou coordinates and colours to Ether Dream units.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn raw_point_to_dac(point: RawPoint) -> DacPoint {
    let position = point.position.map(|value| {
        let normalized = value.clamp(-1.0, 1.0).mul_add(0.5, 0.5);
        normalized.mul_add(65_535.0, -32_768.0) as i16
    });
    let color = point
        .color
        .map(|value| (value.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16);

    DacPoint {
        control: 0,
        x: position[0],
        y: position[1],
        r: color[0],
        g: color[1],
        b: color[2],
        i: 0,
        u1: 0,
        u2: 0,
    }
}

/// Reads the direct-connect override used when WSL cannot receive broadcasts.
fn configured_direct_ip() -> IpAddr {
    let Ok(value) = std::env::var("ETHER_DREAM_IP") else {
        return DEFAULT_ETHER_DREAM_IP;
    };
    value.parse().unwrap_or_else(|error| {
        eprintln!(
            "Ignoring invalid ETHER_DREAM_IP={value:?}: {error}; using {DEFAULT_ETHER_DREAM_IP}"
        );
        DEFAULT_ETHER_DREAM_IP
    })
}

/// Probes a known address and adapts the connection to nannou's DAC descriptor.
fn connect_direct(ip: IpAddr) -> Result<DetectedDac, ether_dream::dac::stream::CommunicationError> {
    let mut broadcast = direct_connection_descriptor();
    let stream = ether_dream::dac::stream::connect_timeout(&broadcast, ip, TCP_TIMEOUT)?;
    broadcast.dac_status = stream.dac().status.to_protocol();
    drop(stream);

    Ok(DetectedDac::EtherDream {
        broadcast,
        source_addr: SocketAddr::new(ip, ether_dream::protocol::BROADCAST_PORT),
    })
}

fn direct_connection_descriptor() -> DacBroadcast {
    // A direct TCP handshake exposes live status but no broadcast metadata.
    // These values match the standard Ether Dream limits used by the device;
    // the subsequent probe replaces the synthetic status before streaming.
    DacBroadcast {
        // A direct TCP handshake does not report the MAC address. The stream is
        // identified by its endpoint instead.
        mac_address: [0; 6],
        hw_revision: 0,
        sw_revision: 2,
        buffer_capacity: ETHER_DREAM_BUFFER_CAPACITY,
        max_point_rate: ETHER_DREAM_MAX_POINT_RATE,
        dac_status: DacStatus {
            protocol: 0,
            light_engine_state: DacStatus::LIGHT_ENGINE_READY,
            playback_state: DacStatus::PLAYBACK_IDLE,
            source: DacStatus::SOURCE_NETWORK_STREAMING,
            light_engine_flags: 0,
            playback_flags: 0,
            source_flags: 0,
            buffer_fullness: 0,
            point_rate: 0,
            point_count: 0,
        },
    }
}

fn dac_address(dac: &DetectedDac) -> String {
    match dac.id() {
        DacId::EtherDream { mac_address } => format!(
            "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
            mac_address[0],
            mac_address[1],
            mac_address[2],
            mac_address[3],
            mac_address[4],
            mac_address[5]
        ),
    }
}

fn set_device(status: &Mutex<EtherDreamStatus>, device: String) {
    lock(status).device = Some(device);
}

fn set_state(status: &Mutex<EtherDreamStatus>, state: ConnectionState) {
    let mut status = lock(status);
    status.state = state;
    status.detail = None;
}

fn set_error(status: &Mutex<EtherDreamStatus>, error: impl std::fmt::Display) {
    let message = error.to_string();
    eprintln!("Ether Dream error: {message}");
    let mut status = lock(status);
    status.state = ConnectionState::Error;
    status.detail = Some(message);
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::{merge_edge_fragments, order_lines_by_travel, resample_line};
    use nannou_laser::Point;

    fn point(x: f32, y: f32) -> Point {
        Point::new([x, y], [1.0, 1.0, 1.0])
    }

    #[test]
    fn nearby_fragments_are_joined_in_continuous_order() {
        let mut lines = vec![
            vec![point(-0.5, 0.0), point(0.0, 0.0)],
            vec![point(0.0, 0.0), point(0.5, 0.0)],
        ];

        merge_edge_fragments(&mut lines);

        assert_eq!(lines.len(), 1);
        let positions: Vec<_> = lines[0].iter().map(|point| point.position).collect();
        assert_eq!(positions, [[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]);
    }

    #[test]
    fn lines_are_oriented_to_reduce_blanked_travel() {
        let lines = vec![
            vec![point(0.9, 0.0), point(0.2, 0.0)],
            vec![point(-0.8, 0.0), point(-0.9, 0.0)],
        ];

        let ordered = order_lines_by_travel(lines);

        assert_eq!(ordered[0][0].position, [0.2, 0.0]);
        assert_eq!(ordered[0][1].position, [0.9, 0.0]);
        assert_eq!(ordered[1][0].position, [-0.8, 0.0]);
    }

    #[test]
    fn resampling_uses_distance_instead_of_vertex_indices() {
        let line = vec![
            point(0.0, 0.0),
            point(0.1, 0.0),
            point(0.2, 0.0),
            point(1.0, 0.0),
        ];

        let sampled = resample_line(&line, 3);

        assert_eq!(sampled.len(), 3);
        assert!((sampled[1].position[0] - 0.5).abs() < 0.000_1);
        assert_eq!(sampled[2].position, [1.0, 0.0]);
    }
}
