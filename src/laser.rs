//! Ether Dream discovery and laser frame streaming.

use std::io::ErrorKind;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use nannou_laser::{DacId, DetectedDac, Frame, Point};

use crate::path_generation::LaserPath;

const DISCOVERY_TIMEOUT: Duration = Duration::from_millis(500);
const TCP_TIMEOUT: Duration = Duration::from_secs(3);

/// A background Ether Dream connection streaming the current laser geometry.
pub struct EtherDreamStream {
    status: Arc<Mutex<String>>,
    stop_tx: mpsc::Sender<()>,
    worker: Option<thread::JoinHandle<()>>,
}

impl EtherDreamStream {
    /// Starts discovering an Ether Dream and streams `path` once one is found.
    pub fn start(path: &LaserPath) -> Self {
        let frame_model = FrameModel {
            points: path.laser_points().to_vec(),
            lines: path.laser_lines().to_vec(),
        };
        let status = Arc::new(Mutex::new(String::from("searching")));
        let worker_status = Arc::clone(&status);
        let (stop_tx, stop_rx) = mpsc::channel();
        let worker = thread::spawn(move || run(frame_model, worker_status, stop_rx));

        Self {
            status,
            stop_tx,
            worker: Some(worker),
        }
    }

    /// Returns the latest discovery or stream status for display.
    pub fn status(&self) -> String {
        self.status
            .lock()
            .map(|status| status.clone())
            .unwrap_or_else(|_| String::from("status unavailable"))
    }
}

impl Drop for EtherDreamStream {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct FrameModel {
    points: Vec<Point>,
    lines: Vec<Vec<Point>>,
}

fn run(frame_model: FrameModel, status: Arc<Mutex<String>>, stop_rx: mpsc::Receiver<()>) {
    let api = nannou_laser::Api::new();
    let mut detected_dacs = match api.detect_dacs() {
        Ok(detected_dacs) => detected_dacs,
        Err(error) => return set_error(&status, error),
    };
    if let Err(error) = detected_dacs.set_timeout(Some(DISCOVERY_TIMEOUT)) {
        return set_error(&status, error);
    }

    // Timed discovery polling keeps shutdown observable while no DAC is present.
    let dac = loop {
        if stop_rx.try_recv().is_ok() {
            return;
        }
        match detected_dacs.next() {
            Some(Ok(dac)) => break dac,
            Some(Err(error))
                if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Some(Err(error)) => return set_error(&status, error),
            None => return set_status(&status, "discovery stopped"),
        }
    };

    let dac_name = dac_name(&dac);
    set_status(&status, format!("connecting to {dac_name}"));
    let stream = match api
        .new_frame_stream(frame_model, render_frame)
        .detected_dac(dac)
        .tcp_timeout(Some(TCP_TIMEOUT))
        .build()
    {
        Ok(stream) => stream,
        Err(error) => return set_error(&status, error),
    };

    set_status(&status, format!("streaming to {dac_name}"));
    // The stream owns transmission until the application requests shutdown.
    let _ = stop_rx.recv();
    drop(stream);
}

fn render_frame(model: &mut FrameModel, frame: &mut Frame) {
    frame.add_points(model.points.iter().copied());
    for line in &model.lines {
        frame.add_lines(line.iter().copied());
    }
}

fn dac_name(dac: &DetectedDac) -> String {
    match dac.id() {
        DacId::EtherDream { mac_address } => format!(
            "Ether Dream {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
            mac_address[0],
            mac_address[1],
            mac_address[2],
            mac_address[3],
            mac_address[4],
            mac_address[5]
        ),
    }
}

fn set_error(status: &Arc<Mutex<String>>, error: impl std::fmt::Display) {
    let message = format!("error: {error}");
    eprintln!("Ether Dream {message}");
    set_status(status, message);
}

fn set_status(status: &Arc<Mutex<String>>, value: impl Into<String>) {
    if let Ok(mut status) = status.lock() {
        *status = value.into();
    }
}
