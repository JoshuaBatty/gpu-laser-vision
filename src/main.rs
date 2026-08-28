//! Real-time CUDA and YOLO vision demo for laser path generation.
//!
//! The binary coordinates video playback, persistent GPU pipelines, dashboard
//! state, path generation, and Ether Dream streaming.

mod cuda_graph;
mod edge_detection;
mod interface;
mod kernels;
mod laser;
mod path_generation;
mod yolo;

use bevy::app::{App as BevyApp, Plugin, PostUpdate};
use bevy::camera::visibility::RenderLayers;
use bevy::window::{
    CursorOptions, Monitor, MonitorSelection, PresentMode, PrimaryWindow, Window, WindowPosition,
    WindowResolution,
};
use nannou::prelude::bevy_asset::{Assets, RenderAssetUsages};
use nannou::prelude::*;
use std::{
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

const SOURCE_IMAGE: &str = "assets/test_tiles.png";
const VIDEO_ASSET: &str = "jcvd_green_screen_720p.mp4";
const PRESENTATION_WIDTH: u32 = 1280;
const PRESENTATION_HEIGHT: u32 = 720;
const PRESENTATION_ASPECT: f32 = 16.0 / 9.0;
const PROJECTOR_RENDER_LAYER: usize = 31;
type AppModel = Result<Model, String>;

/// Long-lived application state shared by the renderer and video pipeline.
struct Model {
    cuda_laser_path: path_generation::LaserPath,
    laser: laser::EtherDreamStream,
    video: Arc<Mutex<VideoBridgeState>>,
    video_label: &'static str,
}

#[derive(Clone)]
struct ImagePanel {
    image: Handle<Image>,
}

#[derive(Clone, Copy)]
struct EdgeThresholds {
    min: f32,
    max: f32,
}

#[derive(Clone, Copy, Default, Eq, PartialEq)]
enum LaserSource {
    #[default]
    Cuda,
    Yolo,
}

impl LaserSource {
    const fn label(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA contour",
            Self::Yolo => "YOLO contour",
        }
    }

    const fn frame_profile(self) -> laser::FrameProfile {
        match self {
            Self::Cuda => laser::FrameProfile::DenseEdges,
            Self::Yolo => laser::FrameProfile::Contour,
        }
    }
}

#[derive(Default)]
struct InterfaceActions {
    toggle_projector: bool,
    laser_changed: bool,
}

#[derive(Default)]
struct OutputState {
    projector: Option<ProjectorOutput>,
    laser_enabled: bool,
    laser_source: LaserSource,
}

/// Entity pair owned by the optional isolated projector output.
#[derive(Clone, Copy)]
struct ProjectorOutput {
    window: Entity,
    camera: Entity,
}

/// Timings for one frame, an accumulated window, or its averaged snapshot.
#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Default)]
struct PipelineTimingSample {
    frame_copy_ms: f64,
    yolo_ms: f64,
    cuda_edges_ms: f64,
    cuda_path_ms: f64,
    yolo_path_ms: f64,
    debug_upload_ms: f64,
    total_ms: f64,
}

impl std::ops::AddAssign for PipelineTimingSample {
    fn add_assign(&mut self, sample: Self) {
        self.frame_copy_ms += sample.frame_copy_ms;
        self.yolo_ms += sample.yolo_ms;
        self.cuda_edges_ms += sample.cuda_edges_ms;
        self.cuda_path_ms += sample.cuda_path_ms;
        self.yolo_path_ms += sample.yolo_path_ms;
        self.debug_upload_ms += sample.debug_upload_ms;
        self.total_ms += sample.total_ms;
    }
}

impl PipelineTimingSample {
    fn averaged_over(self, frames: u32) -> Self {
        let frames: f64 = frames.max(1).into();
        Self {
            frame_copy_ms: self.frame_copy_ms / frames,
            yolo_ms: self.yolo_ms / frames,
            cuda_edges_ms: self.cuda_edges_ms / frames,
            cuda_path_ms: self.cuda_path_ms / frames,
            yolo_path_ms: self.yolo_path_ms / frames,
            debug_upload_ms: self.debug_upload_ms / frames,
            total_ms: self.total_ms / frames,
        }
    }
}

/// Stable metrics snapshot displayed by the dashboard.
#[derive(Clone, Copy, Default)]
struct PipelineMetrics {
    fps: f64,
    timings: PipelineTimingSample,
    yolo_confidence: Option<f32>,
}

impl Default for EdgeThresholds {
    fn default() -> Self {
        Self {
            min: edge_detection::DEFAULT_MIN_THRESHOLD,
            max: edge_detection::DEFAULT_MAX_THRESHOLD,
        }
    }
}

/// Connects Bevy's video entity to state consumed by the nannou view.
#[derive(Component, Clone)]
struct VideoBridge {
    state: Arc<Mutex<VideoBridgeState>>,
    laser: laser::LaserControl,
}

/// Latest video outputs and metrics published across the Bevy/nannou boundary.
#[derive(Default)]
struct VideoBridgeState {
    image: Option<Handle<Image>>,
    processed: Option<ProcessedVideoFrame>,
    error: Option<String>,
    window_sized: bool,
    fps_window_started: Option<Instant>,
    processed_frames: u32,
    timing_totals: PipelineTimingSample,
    metrics: PipelineMetrics,
    interface_configured: bool,
    reported_first_frame: bool,
    thresholds: EdgeThresholds,
    output: OutputState,
}

/// GPU resources reused across video frames.
#[derive(Default)]
struct VideoVisionPipeline {
    detector: Option<edge_detection::CudaEdgeDetector>,
    yolo: Option<yolo::YoloSegmenter>,
}

/// Display textures and laser paths produced for the latest video frame.
struct ProcessedVideoFrame {
    projector: ImagePanel,
    panels: [ImagePanel; 6],
    cuda_laser_path: path_generation::LaserPath,
    yolo_laser_path: path_generation::LaserPath,
}

/// Installs live video processing into Bevy's post-update schedule.
struct VideoBridgePlugin;

impl Plugin for VideoBridgePlugin {
    fn build(&self, app: &mut BevyApp) {
        app.insert_non_send(VideoVisionPipeline::default())
            .add_systems(PostUpdate, (process_video_frames, cleanup_closed_projector));
    }
}

fn main() {
    nannou::app(model)
        .add_plugin(VideoBridgePlugin)
        .view(view)
        .run();
}

fn model(app: &App) -> AppModel {
    app.new_window::<AppModel>()
        .size(PRESENTATION_WIDTH, PRESENTATION_HEIGHT)
        .title("GPU Laser Vision")
        .monitor(MonitorSelection::Primary)
        .primary()
        .build();

    let images = edge_detection::process(SOURCE_IMAGE).map_err(|error| {
        let error = format!("{error:#}");
        eprintln!("Edge detection failed: {error}");
        error
    })?;
    let cuda_laser_path = path_generation::from_edge_mask(
        &images.laser_edges,
        &images.edge_colors,
        &images.edge_pixels,
    );
    let laser = laser::EtherDreamStream::start();
    let video = Arc::new(Mutex::new(VideoBridgeState::default()));
    let video_asset = app.asset_server().load(VIDEO_ASSET);
    app.command_scope({
        let video = video.clone();
        let laser = laser.control().clone();
        move |mut commands| {
            commands.spawn((
                VideoPlayer::new(video_asset).with_mode(PlaybackMode::Loop),
                VideoBridge {
                    state: video,
                    laser,
                },
            ));
        }
    });

    Ok(Model {
        cuda_laser_path,
        laser,
        video,
        video_label: "JCVD green screen - 720p",
    })
}

/// Processes each newly decoded frame through YOLO, CUDA, and path generation.
fn process_video_frames(
    outputs: Query<(&VideoOutput, &VideoBridge), Changed<VideoOutput>>,
    mut assets: ResMut<Assets<Image>>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut pipeline: NonSendMut<VideoVisionPipeline>,
) {
    for (output, bridge) in &outputs {
        publish_source_frame(output, bridge, &mut windows);

        let frame_started = Instant::now();
        let frame_copy_started = Instant::now();
        let frame = {
            let Some(image) = assets.get(&output.image) else {
                continue;
            };
            let Some(pixels) = image.data.as_ref() else {
                continue;
            };
            image::RgbaImage::from_raw(output.size.x, output.size.y, pixels.clone())
        };
        let Some(mut frame) = frame else {
            lock_video(&bridge.state).error = Some("video frame dimensions are invalid".into());
            continue;
        };
        let frame_copy_ms = frame_copy_started.elapsed().as_secs_f64() * 1_000.0;

        // Load the fixed YOLO model once.
        if pipeline.yolo.is_none() {
            match yolo::YoloSegmenter::load(yolo::DEFAULT_MODEL_PATH) {
                Ok(segmenter) => pipeline.yolo = Some(segmenter),
                Err(error) => {
                    lock_video(&bridge.state).error = Some(format!("YOLO setup failed: {error:#}"));
                    continue;
                }
            }
        }
        let yolo_started = Instant::now();
        let yolo_frame = match pipeline
            .yolo
            .as_mut()
            .expect("YOLO was initialized")
            .infer(&frame)
        {
            Ok(frame) => frame,
            Err(error) => {
                lock_video(&bridge.state).error = Some(format!("YOLO inference failed: {error:#}"));
                continue;
            }
        };
        let yolo_ms = yolo_started.elapsed().as_secs_f64() * 1_000.0;

        let dimensions = (frame.width(), frame.height());
        // Rebuild the captured CUDA graph only when frame dimensions change.
        if pipeline
            .detector
            .as_ref()
            .map(edge_detection::CudaEdgeDetector::dimensions)
            != Some(dimensions)
        {
            match edge_detection::CudaEdgeDetector::new(dimensions.0, dimensions.1) {
                Ok(detector) => pipeline.detector = Some(detector),
                Err(error) => {
                    lock_video(&bridge.state).error =
                        Some(format!("CUDA graph setup failed: {error:#}"));
                    continue;
                }
            }
        }

        let detector = pipeline
            .detector
            .as_mut()
            .expect("detector was initialized");
        let thresholds = lock_video(&bridge.state).thresholds;
        let cuda_edges_started = Instant::now();
        let cuda_images = detector.process(&frame, thresholds.min, thresholds.max);
        let cuda_edges_ms = cuda_edges_started.elapsed().as_secs_f64() * 1_000.0;
        match cuda_images {
            Ok(images) => {
                yolo::isolate_person(&mut frame, &yolo_frame.person_mask);
                publish_processed_frame(
                    bridge,
                    &mut assets,
                    images,
                    yolo_frame,
                    frame,
                    PipelineTimingSample {
                        frame_copy_ms,
                        yolo_ms,
                        cuda_edges_ms,
                        ..PipelineTimingSample::default()
                    },
                    frame_started,
                );
            }
            Err(error) => {
                lock_video(&bridge.state).error = Some(format!("CUDA frame failed: {error:#}"));
            }
        }
    }
}

fn publish_source_frame(
    output: &VideoOutput,
    bridge: &VideoBridge,
    windows: &mut Query<&mut Window, With<PrimaryWindow>>,
) {
    let mut video = lock_video(&bridge.state);
    video.image = Some(output.image.clone());
    if !video.window_sized
        && let Ok(mut window) = windows.single_mut()
    {
        let source_size = output.size.as_vec2();
        window.resolution.set(source_size.x, source_size.y);
        video.window_sized = true;
    }
}

/// Publishes one coherent set of textures, paths, and timing data to the UI.
fn publish_processed_frame(
    bridge: &VideoBridge,
    assets: &mut Assets<Image>,
    images: edge_detection::EdgeDetectionImages,
    yolo_frame: yolo::YoloFrame,
    projector_frame: image::RgbaImage,
    mut timings: PipelineTimingSample,
    frame_started: Instant,
) {
    let yolo_confidence = yolo_frame.confidence;
    let cuda_path_started = Instant::now();
    let cuda_laser_path = path_generation::from_edge_mask(
        &images.laser_edges,
        &images.edge_colors,
        &images.edge_pixels,
    );
    timings.cuda_path_ms = cuda_path_started.elapsed().as_secs_f64() * 1_000.0;

    let yolo_path_started = Instant::now();
    let yolo_laser_path = path_generation::from_edge_mask(
        &yolo_frame.contour,
        &yolo_frame.colored_contour,
        &yolo_frame.contour_pixels,
    );
    timings.yolo_path_ms = yolo_path_started.elapsed().as_secs_f64() * 1_000.0;

    // Publish one coherent frame before rolling the metrics window.
    let mut video = lock_video(&bridge.state);
    if video.output.laser_enabled {
        let source = video.output.laser_source;
        let path = match source {
            LaserSource::Cuda => &cuda_laser_path,
            LaserSource::Yolo => &yolo_laser_path,
        };
        bridge.laser.set_path(path, source.frame_profile());
    }
    let cuda_point_count = cuda_laser_path.point_count();
    let yolo_point_count = yolo_laser_path.point_count();
    let debug_upload_started = Instant::now();
    if let Some(processed) = &mut video.processed {
        update_processed_frame(
            processed,
            images,
            yolo_frame,
            projector_frame,
            cuda_laser_path,
            yolo_laser_path,
            assets,
        );
    } else {
        let projector = ImagePanel {
            image: assets.add(Image::from_dynamic(
                image::DynamicImage::ImageRgba8(projector_frame),
                true,
                RenderAssetUsages::default(),
            )),
        };
        let panels = live_image_panels(images, yolo_frame, |image| {
            assets.add(Image::from_dynamic(
                image,
                true,
                RenderAssetUsages::default(),
            ))
        });
        video.processed = Some(ProcessedVideoFrame {
            projector,
            panels,
            cuda_laser_path,
            yolo_laser_path,
        });
    }
    timings.debug_upload_ms = debug_upload_started.elapsed().as_secs_f64() * 1_000.0;
    timings.total_ms = frame_started.elapsed().as_secs_f64() * 1_000.0;
    video.timing_totals += timings;
    video.error = None;
    record_pipeline_metrics(
        &mut video,
        timings,
        yolo_confidence,
        cuda_point_count,
        yolo_point_count,
    );
}

/// Rolls per-frame timings into the dashboard's two-second metrics window.
fn record_pipeline_metrics(
    video: &mut VideoBridgeState,
    timings: PipelineTimingSample,
    yolo_confidence: Option<f32>,
    cuda_point_count: usize,
    yolo_point_count: usize,
) {
    if !video.reported_first_frame {
        let detection = detection_label(yolo_confidence);
        println!(
            "First frame: {:.1} ms ({:.1} ms YOLO, {detection}) | CUDA {cuda_point_count} points | YOLO {yolo_point_count} points",
            timings.total_ms, timings.yolo_ms,
        );
        video.reported_first_frame = true;
    }

    let now = Instant::now();
    let started = *video.fps_window_started.get_or_insert(now);
    video.processed_frames += 1;
    let elapsed = now.duration_since(started).as_secs_f64();
    if elapsed < 2.0 {
        return;
    }

    let processed_frames: f64 = video.processed_frames.into();
    let detection = detection_label(yolo_confidence);
    println!(
        "Pipeline: {:.1} FPS | {:.1} ms YOLO ({detection}) | CUDA {cuda_point_count} points | YOLO {yolo_point_count} points",
        processed_frames / elapsed,
        timings.yolo_ms,
    );
    let average = video.timing_totals.averaged_over(video.processed_frames);
    video.metrics = PipelineMetrics {
        fps: processed_frames / elapsed,
        timings: average,
        yolo_confidence,
    };
    println!(
        "Stages: {:.1} total | {:.1} YOLO | {:.1} CUDA edges | {:.1} CUDA path | {:.1} YOLO path | {:.1} frame copy | {:.1} debug upload ms",
        average.total_ms,
        average.yolo_ms,
        average.cuda_edges_ms,
        average.cuda_path_ms,
        average.yolo_path_ms,
        average.frame_copy_ms,
        average.debug_upload_ms,
    );
    video.fps_window_started = Some(now);
    video.processed_frames = 0;
    video.timing_totals = PipelineTimingSample::default();
}

fn detection_label(confidence: Option<f32>) -> String {
    confidence.map_or_else(
        || "no person".into(),
        |confidence| format!("person {confidence:.2}"),
    )
}

fn update_processed_frame(
    processed: &mut ProcessedVideoFrame,
    images: edge_detection::EdgeDetectionImages,
    yolo_frame: yolo::YoloFrame,
    projector_frame: image::RgbaImage,
    cuda_laser_path: path_generation::LaserPath,
    yolo_laser_path: path_generation::LaserPath,
    assets: &mut Assets<Image>,
) {
    processed.cuda_laser_path = cuda_laser_path;
    processed.yolo_laser_path = yolo_laser_path;
    let previews = images.previews;
    update_rgba_panel(&mut processed.projector, projector_frame, assets);

    let [
        cuda_contour,
        yolo_contour,
        yolo_mask,
        grayscale,
        edges,
        laser_edges,
    ] = &mut processed.panels;
    update_rgb_panel(cuda_contour, images.edge_colors, assets);
    update_rgb_panel(yolo_contour, yolo_frame.colored_contour, assets);
    update_luma_panel(yolo_mask, yolo_frame.person_mask, assets);
    update_luma_panel(grayscale, previews.grayscale, assets);
    update_luma_panel(edges, previews.edges, assets);
    update_luma_panel(laser_edges, images.laser_edges, assets);
}

/// Updates an RGBA preview in place, replacing the asset only if its handle is stale.
fn update_rgba_panel(panel: &mut ImagePanel, image: image::RgbaImage, assets: &mut Assets<Image>) {
    if let Some(mut current) = assets.get_mut(&panel.image) {
        let source = image.into_raw();
        let target = current.data.get_or_insert_with(Vec::new);
        target.clone_from(&source);
        return;
    }

    panel.image = assets.add(Image::from_dynamic(
        image::DynamicImage::ImageRgba8(image),
        true,
        RenderAssetUsages::default(),
    ));
}

/// Updates an RGB preview in place, replacing the asset only if its handle is stale.
fn update_rgb_panel(panel: &mut ImagePanel, image: image::RgbImage, assets: &mut Assets<Image>) {
    if let Some(mut current) = assets.get_mut(&panel.image) {
        let source = image.into_raw();
        let target = current.data.get_or_insert_with(Vec::new);
        target.resize(source.len() / 3 * 4, 255);
        for (source, target) in source.chunks_exact(3).zip(target.chunks_exact_mut(4)) {
            target[..3].copy_from_slice(source);
            target[3] = 255;
        }
        return;
    }

    panel.image = assets.add(Image::from_dynamic(
        image::DynamicImage::ImageRgb8(image),
        true,
        RenderAssetUsages::default(),
    ));
}

/// Updates a grayscale preview in place, replacing the asset only if its handle is stale.
fn update_luma_panel(panel: &mut ImagePanel, image: image::GrayImage, assets: &mut Assets<Image>) {
    if let Some(mut current) = assets.get_mut(&panel.image) {
        let source = image.into_raw();
        let target = current.data.get_or_insert_with(Vec::new);
        target.resize(source.len() * 4, 255);
        for (&source, target) in source.iter().zip(target.chunks_exact_mut(4)) {
            target.fill(source);
            target[3] = 255;
        }
        return;
    }

    panel.image = assets.add(Image::from_dynamic(
        image::DynamicImage::ImageLuma8(image),
        true,
        RenderAssetUsages::default(),
    ));
}

fn live_image_panels(
    images: edge_detection::EdgeDetectionImages,
    yolo_frame: yolo::YoloFrame,
    mut upload: impl FnMut(image::DynamicImage) -> Handle<Image>,
) -> [ImagePanel; 6] {
    let previews = images.previews;
    [
        ImagePanel {
            image: upload(image::DynamicImage::ImageRgb8(images.edge_colors)),
        },
        ImagePanel {
            image: upload(image::DynamicImage::ImageRgb8(yolo_frame.colored_contour)),
        },
        ImagePanel {
            image: upload(image::DynamicImage::ImageLuma8(yolo_frame.person_mask)),
        },
        ImagePanel {
            image: upload(image::DynamicImage::ImageLuma8(previews.grayscale)),
        },
        ImagePanel {
            image: upload(image::DynamicImage::ImageLuma8(previews.edges)),
        },
        ImagePanel {
            image: upload(image::DynamicImage::ImageLuma8(images.laser_edges)),
        },
    ]
}

fn lock_video(video: &Mutex<VideoBridgeState>) -> MutexGuard<'_, VideoBridgeState> {
    video
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn view(app: &App, model: &AppModel, window_entity: Entity) {
    if window_entity != app.main_window().id() {
        return;
    }
    let draw = app.draw_for_window(window_entity);
    let window = app.window(window_entity).rect();
    draw.background().color(interface::background());

    let model = match model {
        Ok(model) => model,
        Err(error) => {
            draw.text("Edge detection failed")
                .x_y(0.0, 20.0)
                .font_size(24)
                .color(Color::srgb_u8(255, 110, 100));
            draw.text(error)
                .x_y(0.0, -20.0)
                .font_size(15)
                .color(Color::srgb_u8(220, 220, 220));
            return;
        }
    };
    let mut video = lock_video(&model.video);
    let laser_status = model.laser.status();
    let laser_label = laser_output_label(
        video.output.laser_enabled,
        video.output.laser_source,
        &laser_status,
    );
    let laser_accent = laser_output_accent(
        video.output.laser_enabled,
        video.output.laser_source,
        &laser_status,
    );
    let egui_context = app.egui_for_window(window_entity);
    if !video.interface_configured {
        interface::configure_egui(&egui_context);
        video.interface_configured = true;
    }
    let mut egui_viewport = egui::Ui::new(
        egui_context.clone(),
        "dashboard_viewport".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(egui_context.viewport_rect()),
    );

    egui::Panel::top("app_header")
        .exact_size(interface::HEADER_HEIGHT)
        .resizable(false)
        .frame(interface::header_frame())
        .show_inside(&mut egui_viewport, |ui| {
            draw_header(ui, video.metrics, &laser_label, laser_accent);
        });

    let mut actions = InterfaceActions::default();
    egui::Panel::left("control_rail")
        .exact_size(interface::SIDEBAR_WIDTH)
        .resizable(false)
        .frame(interface::sidebar_frame())
        .show_inside(&mut egui_viewport, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                actions = draw_control_rail(ui, &mut video, model.video_label, &laser_status);
            });
        });

    let layout = interface::DashboardLayout::new(window);
    draw_source_card(&draw, layout.hero[0], &video);
    if let Some(frame) = video.processed.as_ref() {
        draw_processed_dashboard(&draw, &layout, video.metrics, frame);
    } else {
        draw_loading_dashboard(&draw, &layout, &model.cuda_laser_path);
    }

    if actions.laser_changed {
        apply_laser_settings(model.laser.control(), &video);
    }
    drop(video);
    if actions.toggle_projector {
        toggle_projector_window(app, &model.video);
    }
}

/// Draws only the isolated person texture, cover-cropped to the projector.
fn projector_view(app: &App, model: &AppModel) {
    let Ok(model) = model else {
        return;
    };
    let video = lock_video(&model.video);
    let Some(window_entity) = video.output.projector.map(|projector| projector.window) else {
        return;
    };
    let Some(image) = video
        .processed
        .as_ref()
        .map(|frame| frame.projector.image.clone())
    else {
        return;
    };
    drop(video);

    let draw = app.draw_for_window(window_entity);
    let window = app.window(window_entity).rect();
    draw.background().color(BLACK);
    let (width, height) = if window.w() / window.h() > PRESENTATION_ASPECT {
        (window.w(), window.w() / PRESENTATION_ASPECT)
    } else {
        (window.h() * PRESENTATION_ASPECT, window.h())
    };
    interface::draw_texture(
        &draw,
        &image,
        interface::CardRect {
            x: 0.0,
            y: 0.0,
            width,
            height,
        },
    );
}

/// Creates or destroys the borderless window and camera on the second display.
fn toggle_projector_window(app: &App, video: &Mutex<VideoBridgeState>) {
    if let Some(projector) = lock_video(video).output.projector.take() {
        app.command_scope(move |mut commands| {
            commands.entity(projector.window).try_despawn();
            commands.entity(projector.camera).try_despawn();
        });
        return;
    }

    let Some(monitor) = secondary_monitor(app) else {
        eprintln!("Projector output requires a secondary display");
        return;
    };

    let projector_layer = RenderLayers::layer(PROJECTOR_RENDER_LAYER);
    let camera = app.new_camera().layer(projector_layer.clone()).build();
    let projector = app
        .new_window::<AppModel>()
        .window(Window {
            title: "GPU Laser Vision · Projector".into(),
            position: WindowPosition::At(monitor.physical_position),
            resolution: WindowResolution::new(monitor.physical_width, monitor.physical_height)
                .with_scale_factor_override(1.0),
            present_mode: PresentMode::Mailbox,
            decorations: false,
            resizable: false,
            ..Window::default()
        })
        .camera(camera)
        .view(projector_view)
        .build();
    app.command_scope(move |mut commands| {
        commands.entity(projector).insert((
            CursorOptions {
                visible: false,
                ..CursorOptions::default()
            },
            projector_layer.clone(),
        ));
        commands.entity(camera).insert(projector_layer);
    });
    lock_video(video).output.projector = Some(ProjectorOutput {
        window: projector,
        camera,
    });
}

/// Selects the first monitor that is not Bevy's primary monitor.
fn secondary_monitor(app: &App) -> Option<Monitor> {
    let primary = app.primary_monitor();
    let mut monitors = app.available_monitors().into_iter();
    match primary {
        Some(primary) => monitors
            .find(|(entity, _)| *entity != primary)
            .map(|(_, monitor)| monitor),
        None => monitors.nth(1).map(|(_, monitor)| monitor),
    }
}

/// Removes projector state and its camera after the window closes externally.
fn cleanup_closed_projector(
    bridges: Query<&VideoBridge>,
    windows: Query<(), With<Window>>,
    mut commands: Commands,
) {
    for bridge in &bridges {
        let mut video = lock_video(&bridge.state);
        let Some(projector) = video
            .output
            .projector
            .filter(|projector| windows.get(projector.window).is_err())
        else {
            continue;
        };

        video.output.projector = None;
        commands.entity(projector.camera).try_despawn();
    }
}

/// Installs geometry before enabling output so the worker never exposes stale data.
fn apply_laser_settings(control: &laser::LaserControl, video: &VideoBridgeState) {
    if video.output.laser_enabled
        && let Some(frame) = &video.processed
    {
        let source = video.output.laser_source;
        let path = match source {
            LaserSource::Cuda => &frame.cuda_laser_path,
            LaserSource::Yolo => &frame.yolo_laser_path,
        };
        control.set_path(path, source.frame_profile());
    }
    control.set_enabled(video.output.laser_enabled);
}

fn draw_source_card(draw: &Draw, card: interface::CardRect, video: &VideoBridgeState) {
    let source_meta = if video.error.is_some() {
        "SOURCE ERROR"
    } else if video.image.is_some() {
        "LIVE · 1280 × 720"
    } else {
        "CONNECTING"
    };
    let source_media = interface::draw_card_shell(
        draw,
        card,
        "SOURCE",
        "INPUT VIDEO",
        source_meta,
        if video.error.is_some() {
            interface::Accent::Error
        } else {
            interface::Accent::Neutral
        },
    );
    if let Some(image) = &video.image {
        interface::draw_texture(draw, image, source_media);
    } else {
        interface::draw_empty_state(draw, source_media, "WAITING FOR VIDEO");
    }
}

fn draw_processed_dashboard(
    draw: &Draw,
    layout: &interface::DashboardLayout,
    metrics: PipelineMetrics,
    frame: &ProcessedVideoFrame,
) {
    let [
        cuda_contour,
        yolo_contour,
        yolo_mask,
        grayscale,
        scharr,
        cuda_mask,
    ] = &frame.panels;

    let cuda_contour_media = interface::draw_card_shell(
        draw,
        layout.hero[1],
        "CUDA",
        "COLORED CONTOUR",
        &format!("{} PTS", frame.cuda_laser_path.point_count()),
        interface::Accent::Cuda,
    );
    interface::draw_texture(draw, &cuda_contour.image, cuda_contour_media);

    let confidence = metrics.yolo_confidence.map_or_else(
        || "SEARCHING".into(),
        |value| format!("{:.0}% PERSON", value * 100.0),
    );
    let yolo_contour_media = interface::draw_card_shell(
        draw,
        layout.hero[2],
        "YOLO",
        "COLORED CONTOUR",
        &confidence,
        interface::Accent::Yolo,
    );
    interface::draw_texture(draw, &yolo_contour.image, yolo_contour_media);

    let cuda_laser_meta = format!(
        "{} LINES · {} PTS",
        frame.cuda_laser_path.laser_lines().len(),
        frame.cuda_laser_path.point_count()
    );
    let cuda_laser_media = interface::draw_card_shell(
        draw,
        layout.outputs[0],
        "CUDA OUTPUT",
        "LASER PREVIEW",
        &cuda_laser_meta,
        interface::Accent::Cuda,
    );
    draw_laser_path(draw, &frame.cuda_laser_path, cuda_laser_media);

    let yolo_laser_meta = format!(
        "{} LINES · {} PTS",
        frame.yolo_laser_path.laser_lines().len(),
        frame.yolo_laser_path.point_count()
    );
    let yolo_laser_media = interface::draw_card_shell(
        draw,
        layout.outputs[1],
        "YOLO OUTPUT",
        "LASER PREVIEW",
        &yolo_laser_meta,
        interface::Accent::Yolo,
    );
    draw_laser_path(draw, &frame.yolo_laser_path, yolo_laser_media);

    for (card, eyebrow, title, panel) in [
        (layout.diagnostics[0], "CUDA STAGE", "GRAYSCALE", grayscale),
        (layout.diagnostics[1], "CUDA STAGE", "SCHARR", scharr),
        (layout.diagnostics[2], "CUDA STAGE", "EDGE MASK", cuda_mask),
        (
            layout.diagnostics[3],
            "YOLO STAGE",
            "PERSON MASK",
            yolo_mask,
        ),
    ] {
        let media = interface::draw_card_shell(
            draw,
            card,
            eyebrow,
            title,
            "",
            interface::Accent::Diagnostic,
        );
        interface::draw_texture(draw, &panel.image, media);
    }
}

fn draw_header(
    ui: &mut egui::Ui,
    metrics: PipelineMetrics,
    laser_status: &str,
    laser_accent: interface::Accent,
) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(
                egui::RichText::new("GPU LASER VISION")
                    .size(18.0)
                    .strong()
                    .color(egui::Color32::from_rgb(236, 240, 242)),
            );
            ui.label(
                egui::RichText::new("REAL-TIME CUDA + YOLO SEGMENTATION PIPELINE")
                    .size(9.0)
                    .strong()
                    .color(egui::Color32::from_rgb(105, 119, 130)),
            );
        });

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            interface::status_chip(ui, "LASER", laser_status, laser_accent);
            let yolo = metrics.yolo_confidence.map_or_else(
                || "WARMING".into(),
                |value| format!("{:.0}%", value * 100.0),
            );
            interface::status_chip(ui, "YOLO", &yolo, interface::Accent::Yolo);
            interface::status_chip(ui, "CUDA", "GRAPH", interface::Accent::Cuda);
            let fps = if metrics.fps > 0.0 {
                format!("{:.0} FPS", metrics.fps)
            } else {
                "-- FPS".into()
            };
            interface::status_chip(ui, "PIPELINE", &fps, interface::Accent::Neutral);
        });
    });
}

fn draw_control_rail(
    ui: &mut egui::Ui,
    video: &mut VideoBridgeState,
    video_label: &str,
    laser_status: &laser::EtherDreamStatus,
) -> InterfaceActions {
    let mut actions = InterfaceActions::default();
    draw_threshold_controls(ui, &mut video.thresholds);

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(10.0);
    draw_output_controls(ui, video, laser_status, &mut actions);

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(10.0);
    interface::section_label(ui, "PIPELINE");
    ui.add_space(4.0);
    let ready = video.processed.is_some();
    interface::metric_row(
        ui,
        "CUDA GRAPH",
        state_text(
            if ready { "READY" } else { "WARMING" },
            interface::Accent::Cuda,
        ),
    );
    interface::metric_row(
        ui,
        "YOLO RTX",
        state_text(
            if ready { "ACTIVE" } else { "LOADING" },
            interface::Accent::Yolo,
        ),
    );
    interface::metric_row(
        ui,
        "ETHER DREAM",
        state_text(
            laser_status.state().label(),
            connection_accent(laser_status),
        ),
    );

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(10.0);
    interface::section_label(ui, "PERFORMANCE");
    ui.add_space(4.0);
    let timings = video.metrics.timings;
    interface::metric_row(ui, "FRAME RATE", format_fps(video.metrics.fps));
    interface::metric_row(ui, "TOTAL", format_ms(timings.total_ms));
    interface::metric_row(ui, "YOLO", format_ms(timings.yolo_ms));
    interface::metric_row(ui, "CUDA EDGES", format_ms(timings.cuda_edges_ms));
    interface::metric_row(
        ui,
        "PATH BUILD",
        format_ms(timings.cuda_path_ms + timings.yolo_path_ms),
    );
    interface::metric_row(ui, "TEXTURE UPLOAD", format_ms(timings.debug_upload_ms));

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(10.0);
    interface::section_label(ui, "SOURCE");
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(video_label)
            .size(12.0)
            .color(egui::Color32::from_rgb(188, 197, 203)),
    );
    ui.label(
        egui::RichText::new("1280 × 720 · LOOP")
            .size(10.0)
            .strong()
            .color(egui::Color32::from_rgb(103, 117, 127)),
    );

    if let Some(error) = &video.error {
        ui.add_space(14.0);
        interface::section_label(ui, "PIPELINE ERROR");
        ui.label(
            egui::RichText::new(error)
                .size(10.0)
                .color(interface::Accent::Error.ui_color()),
        );
    }

    actions
}

fn draw_output_controls(
    ui: &mut egui::Ui,
    video: &mut VideoBridgeState,
    laser_status: &laser::EtherDreamStatus,
    actions: &mut InterfaceActions,
) {
    interface::section_label(ui, "OUTPUTS");
    ui.add_space(6.0);

    let projector_open = video.output.projector.is_some();
    let projector_label = if projector_open {
        "Close projector output"
    } else {
        "Open fullscreen projector"
    };
    let projector_button =
        egui::Button::new(projector_label).min_size(egui::vec2(ui.available_width(), 30.0));
    if ui.add(projector_button).clicked() {
        actions.toggle_projector = true;
    }

    ui.add_space(8.0);
    ui.label(
        egui::RichText::new("Laser source")
            .size(11.0)
            .color(egui::Color32::from_rgb(121, 134, 144)),
    );
    let previous_source = video.output.laser_source;
    egui::ComboBox::from_id_salt("laser_source")
        .width(ui.available_width())
        .selected_text(video.output.laser_source.label())
        .show_ui(ui, |ui| {
            ui.selectable_value(
                &mut video.output.laser_source,
                LaserSource::Cuda,
                LaserSource::Cuda.label(),
            );
            ui.selectable_value(
                &mut video.output.laser_source,
                LaserSource::Yolo,
                LaserSource::Yolo.label(),
            );
        });
    actions.laser_changed |= video.output.laser_source != previous_source;

    let laser_label = if video.output.laser_enabled {
        "Disable laser"
    } else {
        "Enable laser"
    };
    let mut laser_button =
        egui::Button::new(laser_label).min_size(egui::vec2(ui.available_width(), 30.0));
    if video.output.laser_enabled {
        laser_button = laser_button.fill(egui::Color32::from_rgb(31, 92, 68));
    }
    if ui.add(laser_button).clicked() {
        video.output.laser_enabled = !video.output.laser_enabled;
        actions.laser_changed = true;
    }

    ui.add_space(6.0);
    interface::metric_row(
        ui,
        "DAC",
        state_text(
            laser_status.device().unwrap_or("NOT DETECTED"),
            connection_accent(laser_status),
        ),
    );
    interface::metric_row(
        ui,
        "CONNECTION",
        state_text(
            laser_status.state().label(),
            connection_accent(laser_status),
        ),
    );
    if let Some(detail) = laser_status.detail() {
        ui.label(
            egui::RichText::new(detail)
                .size(9.0)
                .color(interface::Accent::Error.ui_color()),
        );
    }
}

fn draw_threshold_controls(ui: &mut egui::Ui, thresholds: &mut EdgeThresholds) {
    ui.spacing_mut().item_spacing.y = 4.0;
    interface::section_label(ui, "VISION CONTROLS");
    ui.add_space(3.0);
    ui.label(egui::RichText::new("Edge thresholds").size(16.0).strong());
    ui.label(
        egui::RichText::new("Normalized Scharr magnitude")
            .size(11.0)
            .color(egui::Color32::from_rgb(121, 134, 144)),
    );
    ui.add_space(8.0);

    let mut min = thresholds.min;
    let mut max = thresholds.max;
    interface::metric_row(
        ui,
        "MIN THRESHOLD",
        egui::RichText::new(format!("{min:.3}"))
            .strong()
            .color(interface::Accent::Cuda.ui_color()),
    );
    ui.add(egui::Slider::new(&mut min, 0.0..=max).show_value(false));
    interface::metric_row(
        ui,
        "MAX THRESHOLD",
        egui::RichText::new(format!("{max:.3}"))
            .strong()
            .color(interface::Accent::Cuda.ui_color()),
    );
    ui.add(egui::Slider::new(&mut max, min..=1.0).show_value(false));
    *thresholds = EdgeThresholds { min, max };

    ui.add_space(4.0);
    if ui.button("Reset thresholds").clicked() {
        *thresholds = EdgeThresholds::default();
    }
}

fn draw_loading_dashboard(
    draw: &Draw,
    layout: &interface::DashboardLayout,
    fallback_laser_path: &path_generation::LaserPath,
) {
    for (card, eyebrow, title, accent) in [
        (
            layout.hero[1],
            "CUDA",
            "COLORED CONTOUR",
            interface::Accent::Cuda,
        ),
        (
            layout.hero[2],
            "YOLO",
            "COLORED CONTOUR",
            interface::Accent::Yolo,
        ),
    ] {
        let media = interface::draw_card_shell(draw, card, eyebrow, title, "WARMING", accent);
        interface::draw_empty_state(draw, media, "INITIALIZING PIPELINE");
    }

    let cuda_media = interface::draw_card_shell(
        draw,
        layout.outputs[0],
        "CUDA OUTPUT",
        "LASER PREVIEW",
        "WARMING",
        interface::Accent::Cuda,
    );
    draw_laser_path(draw, fallback_laser_path, cuda_media);
    let yolo_media = interface::draw_card_shell(
        draw,
        layout.outputs[1],
        "YOLO OUTPUT",
        "LASER PREVIEW",
        "WARMING",
        interface::Accent::Yolo,
    );
    interface::draw_empty_state(draw, yolo_media, "AWAITING SEGMENTATION");

    for (card, eyebrow, title) in [
        (layout.diagnostics[0], "CUDA STAGE", "GRAYSCALE"),
        (layout.diagnostics[1], "CUDA STAGE", "SCHARR"),
        (layout.diagnostics[2], "CUDA STAGE", "EDGE MASK"),
        (layout.diagnostics[3], "YOLO STAGE", "PERSON MASK"),
    ] {
        let media = interface::draw_card_shell(
            draw,
            card,
            eyebrow,
            title,
            "",
            interface::Accent::Diagnostic,
        );
        interface::draw_empty_state(draw, media, "WAITING");
    }
}

fn laser_output_label(
    enabled: bool,
    source: LaserSource,
    status: &laser::EtherDreamStatus,
) -> String {
    if enabled && status.state() == laser::ConnectionState::Streaming {
        return match source {
            LaserSource::Cuda => "CUDA ON".into(),
            LaserSource::Yolo => "YOLO ON".into(),
        };
    }
    if enabled {
        return format!("ARMED · {}", status.state().label());
    }
    if status.state() == laser::ConnectionState::Streaming {
        "READY".into()
    } else {
        status.state().label().into()
    }
}

fn laser_output_accent(
    enabled: bool,
    source: LaserSource,
    status: &laser::EtherDreamStatus,
) -> interface::Accent {
    if status.state() == laser::ConnectionState::Error {
        interface::Accent::Error
    } else if enabled {
        match source {
            LaserSource::Cuda => interface::Accent::Cuda,
            LaserSource::Yolo => interface::Accent::Yolo,
        }
    } else {
        connection_accent(status)
    }
}

fn connection_accent(status: &laser::EtherDreamStatus) -> interface::Accent {
    match status.state() {
        laser::ConnectionState::Streaming => interface::Accent::Yolo,
        laser::ConnectionState::Error => interface::Accent::Error,
        _ => interface::Accent::Neutral,
    }
}

fn state_text(value: &str, accent: interface::Accent) -> egui::RichText {
    egui::RichText::new(value.to_ascii_uppercase())
        .size(10.0)
        .strong()
        .color(accent.ui_color())
}

fn format_ms(value: f64) -> String {
    if value > 0.0 {
        format!("{value:.1} ms")
    } else {
        "--".into()
    }
}

fn format_fps(value: f64) -> String {
    if value > 0.0 {
        format!("{value:.1} FPS")
    } else {
        "--".into()
    }
}

fn draw_laser_path(
    draw: &Draw,
    laser_path: &path_generation::LaserPath,
    media: interface::CardRect,
) {
    let x_scale = media.width * 0.5;
    let y_scale = media.height * 0.5;
    for line in laser_path.laser_lines() {
        for segment in line.windows(2) {
            let start = segment[0];
            let end = segment[1];
            let color = [
                (start.color[0] + end.color[0]) * 0.5,
                (start.color[1] + end.color[1]) * 0.5,
                (start.color[2] + end.color[2]) * 0.5,
            ];
            draw.line()
                .start(pt2(
                    media.x + start.position[0] * x_scale,
                    media.y + start.position[1] * y_scale,
                ))
                .end(pt2(
                    media.x + end.position[0] * x_scale,
                    media.y + end.position[1] * y_scale,
                ))
                .weight(1.1)
                .color(Color::srgb(color[0], color[1], color[2]));
        }
    }

    for point in laser_path.laser_points() {
        draw.ellipse()
            .x_y(
                media.x + point.position[0] * x_scale,
                media.y + point.position[1] * y_scale,
            )
            .radius(1.0)
            .color(Color::srgb(point.color[0], point.color[1], point.color[2]));
    }
}
