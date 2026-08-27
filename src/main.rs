mod cuda_graph;
mod edge_detection;
mod kernels;
mod laser;
mod path_generation;
mod yolo;

use bevy::app::{App as BevyApp, Plugin, PostUpdate};
use bevy::window::{PrimaryWindow, Window};
use nannou::prelude::bevy_asset::{Assets, RenderAssetUsages};
use nannou::prelude::*;
use std::{
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

const SOURCE_IMAGE: &str = "assets/test_tiles.png";
const PREFERRED_VIDEO_ASSET: &str = "jcvd_green_screen_720p.mp4";
const FALLBACK_VIDEO_ASSET: &str = "big_buck_bunny_720p.mp4";
const COLUMNS: usize = 3;
const CONTROL_PANEL_WIDTH: f32 = 220.0;

type AppModel = Result<Model, String>;

struct Model {
    panels: Vec<ImagePanel>,
    cuda_laser_path: path_generation::LaserPath,
    laser: laser::EtherDreamStream,
    video: Arc<Mutex<VideoBridgeState>>,
    video_label: &'static str,
}

#[derive(Clone)]
struct ImagePanel {
    label: &'static str,
    image: Handle<Image>,
}

enum Visualization<'a> {
    Video,
    Image(&'a ImagePanel),
    Laser {
        label: &'a str,
        path: &'a path_generation::LaserPath,
    },
}

#[derive(Clone, Copy)]
struct EdgeThresholds {
    min: f32,
    max: f32,
}

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
    fn averaged_over(self, frames: u64) -> Self {
        let frames = frames.max(1) as f64;
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

impl Default for EdgeThresholds {
    fn default() -> Self {
        Self {
            min: edge_detection::DEFAULT_MIN_THRESHOLD,
            max: edge_detection::DEFAULT_MAX_THRESHOLD,
        }
    }
}

#[derive(Component, Clone)]
struct VideoBridge(Arc<Mutex<VideoBridgeState>>);

#[derive(Default)]
struct VideoBridgeState {
    image: Option<Handle<Image>>,
    processed: Option<ProcessedVideoFrame>,
    error: Option<String>,
    window_sized: bool,
    fps_window_started: Option<Instant>,
    processed_frames: u64,
    timing_totals: PipelineTimingSample,
    reported_first_frame: bool,
    thresholds: EdgeThresholds,
}

#[derive(Default)]
struct VideoVisionPipeline {
    detector: Option<edge_detection::CudaEdgeDetector>,
    yolo: Option<yolo::YoloSegmenter>,
}

struct ProcessedVideoFrame {
    panels: [ImagePanel; 6],
    cuda_laser_path: path_generation::LaserPath,
    yolo_laser_path: path_generation::LaserPath,
}

struct VideoBridgePlugin;

impl Plugin for VideoBridgePlugin {
    fn build(&self, app: &mut BevyApp) {
        app.insert_non_send(VideoVisionPipeline::default())
            .add_systems(PostUpdate, process_video_frames);
    }
}

fn main() {
    nannou::app(model)
        .add_plugin(VideoBridgePlugin)
        .view(view)
        .run();
}

fn model(app: &App) -> AppModel {
    let (width, height) = image::image_dimensions(SOURCE_IMAGE).unwrap_or((800, 800));
    app.new_window::<AppModel>()
        .size(width, height)
        .title("GPU Laser Vision")
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
    let laser = laser::EtherDreamStream::start(&cuda_laser_path);
    let video = Arc::new(Mutex::new(VideoBridgeState::default()));
    let (video_asset_name, video_label) = if app.assets_path().join(PREFERRED_VIDEO_ASSET).is_file()
    {
        (PREFERRED_VIDEO_ASSET, "JCVD green screen - 720p")
    } else {
        (FALLBACK_VIDEO_ASSET, "Big Buck Bunny - 720p")
    };
    let video_asset = app.asset_server().load(video_asset_name);
    app.command_scope({
        let video = video.clone();
        move |mut commands| {
            commands.spawn((
                VideoPlayer::new(video_asset).with_mode(PlaybackMode::Loop),
                VideoBridge(video),
            ));
        }
    });

    let upload = |image| {
        let image = Image::from_dynamic(image, true, RenderAssetUsages::default());
        app.asset_server().add(image)
    };

    Ok(Model {
        panels: image_panels(images, upload),
        cuda_laser_path,
        laser,
        video,
        video_label,
    })
}

fn process_video_frames(
    outputs: Query<(&VideoOutput, &VideoBridge), Changed<VideoOutput>>,
    mut assets: ResMut<Assets<Image>>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut pipeline: NonSendMut<VideoVisionPipeline>,
) {
    for (output, bridge) in &outputs {
        {
            let mut video = lock_video(&bridge.0);
            video.image = Some(output.image.clone());
            if !video.window_sized
                && let Ok(mut window) = windows.single_mut()
            {
                window
                    .resolution
                    .set(output.size.x as f32, output.size.y as f32);
                video.window_sized = true;
            }
        }

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
        let Some(frame) = frame else {
            lock_video(&bridge.0).error = Some("video frame dimensions are invalid".into());
            continue;
        };
        let frame_copy_ms = frame_copy_started.elapsed().as_secs_f64() * 1_000.0;

        if pipeline.yolo.is_none() {
            match yolo::YoloSegmenter::load(yolo::DEFAULT_MODEL_PATH) {
                Ok(segmenter) => pipeline.yolo = Some(segmenter),
                Err(error) => {
                    lock_video(&bridge.0).error = Some(format!("YOLO setup failed: {error:#}"));
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
                lock_video(&bridge.0).error = Some(format!("YOLO inference failed: {error:#}"));
                continue;
            }
        };
        let yolo_ms = yolo_started.elapsed().as_secs_f64() * 1_000.0;
        let yolo_confidence = yolo_frame.confidence;

        let dimensions = (frame.width(), frame.height());
        if pipeline
            .detector
            .as_ref()
            .map(|detector| detector.dimensions())
            != Some(dimensions)
        {
            match edge_detection::CudaEdgeDetector::new(dimensions.0, dimensions.1) {
                Ok(detector) => pipeline.detector = Some(detector),
                Err(error) => {
                    lock_video(&bridge.0).error =
                        Some(format!("CUDA graph setup failed: {error:#}"));
                    continue;
                }
            }
        }

        let detector = pipeline
            .detector
            .as_mut()
            .expect("detector was initialized");
        let thresholds = lock_video(&bridge.0).thresholds;
        let cuda_edges_started = Instant::now();
        let cuda_images = detector.process(frame, thresholds.min, thresholds.max);
        let cuda_edges_ms = cuda_edges_started.elapsed().as_secs_f64() * 1_000.0;
        match cuda_images {
            Ok(images) => {
                let cuda_path_started = Instant::now();
                let cuda_laser_path = path_generation::from_edge_mask(
                    &images.laser_edges,
                    &images.edge_colors,
                    &images.edge_pixels,
                );
                let cuda_path_ms = cuda_path_started.elapsed().as_secs_f64() * 1_000.0;
                let yolo_path_started = Instant::now();
                let yolo_laser_path = path_generation::from_edge_mask(
                    &yolo_frame.contour,
                    &yolo_frame.colored_contour,
                    &yolo_frame.contour_pixels,
                );
                let yolo_path_ms = yolo_path_started.elapsed().as_secs_f64() * 1_000.0;
                let mut video = lock_video(&bridge.0);
                let cuda_point_count = cuda_laser_path.point_count();
                let yolo_point_count = yolo_laser_path.point_count();
                let debug_upload_started = Instant::now();
                if let Some(processed) = &mut video.processed {
                    update_processed_frame(
                        processed,
                        images,
                        yolo_frame,
                        cuda_laser_path,
                        yolo_laser_path,
                        &mut assets,
                    );
                } else {
                    let panels = live_image_panels(images, yolo_frame, |image| {
                        assets.add(Image::from_dynamic(
                            image,
                            true,
                            RenderAssetUsages::default(),
                        ))
                    });
                    video.processed = Some(ProcessedVideoFrame {
                        panels,
                        cuda_laser_path,
                        yolo_laser_path,
                    });
                }
                let debug_upload_ms = debug_upload_started.elapsed().as_secs_f64() * 1_000.0;
                let total_ms = frame_started.elapsed().as_secs_f64() * 1_000.0;
                video.timing_totals += PipelineTimingSample {
                    frame_copy_ms,
                    yolo_ms,
                    cuda_edges_ms,
                    cuda_path_ms,
                    yolo_path_ms,
                    debug_upload_ms,
                    total_ms,
                };
                video.error = None;
                if !video.reported_first_frame {
                    let detection = yolo_confidence
                        .map(|confidence| format!("person {confidence:.2}"))
                        .unwrap_or_else(|| "no person".into());
                    println!(
                        "First frame: {:.1} ms ({yolo_ms:.1} ms YOLO, {detection}) | CUDA {cuda_point_count} points | YOLO {yolo_point_count} points",
                        frame_started.elapsed().as_secs_f64() * 1_000.0,
                    );
                    video.reported_first_frame = true;
                }
                let now = Instant::now();
                let started = *video.fps_window_started.get_or_insert(now);
                video.processed_frames += 1;
                let elapsed = now.duration_since(started).as_secs_f64();
                if elapsed >= 2.0 {
                    let detection = yolo_confidence
                        .map(|confidence| format!("person {confidence:.2}"))
                        .unwrap_or_else(|| "no person".into());
                    println!(
                        "Pipeline: {:.1} FPS | {yolo_ms:.1} ms YOLO ({detection}) | CUDA {cuda_point_count} points | YOLO {yolo_point_count} points",
                        video.processed_frames as f64 / elapsed,
                    );
                    let average = video.timing_totals.averaged_over(video.processed_frames);
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
            }
            Err(error) => {
                lock_video(&bridge.0).error = Some(format!("CUDA frame failed: {error:#}"));
            }
        }
    }
}

fn update_processed_frame(
    processed: &mut ProcessedVideoFrame,
    images: edge_detection::EdgeDetectionImages,
    yolo_frame: yolo::YoloFrame,
    cuda_laser_path: path_generation::LaserPath,
    yolo_laser_path: path_generation::LaserPath,
    assets: &mut Assets<Image>,
) {
    processed.cuda_laser_path = cuda_laser_path;
    processed.yolo_laser_path = yolo_laser_path;
    let previews = images.previews;

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

fn image_panels(
    images: edge_detection::EdgeDetectionImages,
    mut upload: impl FnMut(image::DynamicImage) -> Handle<Image>,
) -> Vec<ImagePanel> {
    let previews = images.previews;
    vec![
        ImagePanel {
            label: "Original",
            image: upload(image::DynamicImage::ImageRgba8(images.original)),
        },
        ImagePanel {
            label: "Grayscale",
            image: upload(image::DynamicImage::ImageLuma8(previews.grayscale)),
        },
        ImagePanel {
            label: "Scharr magnitude",
            image: upload(image::DynamicImage::ImageLuma8(previews.edges)),
        },
        ImagePanel {
            label: "Laser edge mask",
            image: upload(image::DynamicImage::ImageLuma8(images.laser_edges)),
        },
        ImagePanel {
            label: "GPU edge colours",
            image: upload(image::DynamicImage::ImageRgb8(images.edge_colors)),
        },
    ]
}

fn live_image_panels(
    images: edge_detection::EdgeDetectionImages,
    yolo_frame: yolo::YoloFrame,
    mut upload: impl FnMut(image::DynamicImage) -> Handle<Image>,
) -> [ImagePanel; 6] {
    let previews = images.previews;
    [
        ImagePanel {
            label: "CUDA colored contour",
            image: upload(image::DynamicImage::ImageRgb8(images.edge_colors)),
        },
        ImagePanel {
            label: "YOLO colored contour",
            image: upload(image::DynamicImage::ImageRgb8(yolo_frame.colored_contour)),
        },
        ImagePanel {
            label: "YOLO person mask",
            image: upload(image::DynamicImage::ImageLuma8(yolo_frame.person_mask)),
        },
        ImagePanel {
            label: "Grayscale",
            image: upload(image::DynamicImage::ImageLuma8(previews.grayscale)),
        },
        ImagePanel {
            label: "Scharr magnitude",
            image: upload(image::DynamicImage::ImageLuma8(previews.edges)),
        },
        ImagePanel {
            label: "CUDA edge mask",
            image: upload(image::DynamicImage::ImageLuma8(images.laser_edges)),
        },
    ]
}

fn lock_video(video: &Mutex<VideoBridgeState>) -> MutexGuard<'_, VideoBridgeState> {
    video
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn view(app: &App, model: &AppModel, window_entity: Entity) {
    let draw = app.draw();
    let window = app.window_rect();
    draw.background().color(Color::srgb_u8(12, 14, 16));

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
    let egui_context = app.egui_for_window(window_entity);
    let mut egui_viewport = egui::Ui::new(
        egui_context.clone(),
        "edge_threshold_viewport".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(egui_context.viewport_rect()),
    );
    egui::Panel::left("edge_threshold_controls")
        .exact_size(CONTROL_PANEL_WIDTH)
        .resizable(false)
        .show_inside(&mut egui_viewport, |ui| {
            ui.heading("Edge thresholds");
            ui.label("Normalized Scharr magnitude");
            ui.add_space(8.0);

            let mut min = video.thresholds.min;
            let mut max = video.thresholds.max;
            ui.add(
                egui::Slider::new(&mut min, 0.0..=max)
                    .text("Min")
                    .fixed_decimals(3),
            );
            ui.add(
                egui::Slider::new(&mut max, min..=1.0)
                    .text("Max")
                    .fixed_decimals(3),
            );
            video.thresholds = EdgeThresholds { min, max };

            if ui.button("Reset").clicked() {
                video.thresholds = EdgeThresholds::default();
            }
        });
    let processed = video.processed.as_ref();
    let cuda_laser_path = processed.map_or(&model.cuda_laser_path, |frame| &frame.cuda_laser_path);
    let cuda_laser_label = format!(
        "CUDA laser - {} lines / {} points - {}",
        cuda_laser_path.laser_lines().len(),
        cuda_laser_path.point_count(),
        model.laser.status()
    );
    let yolo_laser_label = processed.map(|frame| {
        format!(
            "YOLO laser - {} lines / {} points",
            frame.yolo_laser_path.laser_lines().len(),
            frame.yolo_laser_path.point_count()
        )
    });
    let video_label = video.error.as_deref().unwrap_or(model.video_label);
    let mut visualizations = vec![Visualization::Video];

    if let Some(frame) = processed {
        let [
            cuda_contour,
            yolo_contour,
            yolo_mask,
            grayscale,
            scharr,
            cuda_mask,
        ] = &frame.panels;
        visualizations.extend([
            Visualization::Image(cuda_contour),
            Visualization::Image(yolo_contour),
            Visualization::Laser {
                label: &cuda_laser_label,
                path: &frame.cuda_laser_path,
            },
            Visualization::Laser {
                label: yolo_laser_label
                    .as_deref()
                    .expect("live YOLO label was initialized"),
                path: &frame.yolo_laser_path,
            },
            Visualization::Image(yolo_mask),
            Visualization::Image(grayscale),
            Visualization::Image(scharr),
            Visualization::Image(cuda_mask),
        ]);
    } else {
        visualizations.extend(model.panels.iter().map(Visualization::Image));
        visualizations.push(Visualization::Laser {
            label: &cuda_laser_label,
            path: &model.cuda_laser_path,
        });
    }

    let margin = 24.0;
    let gap = 18.0;
    let label_height = 30.0;
    let panel_count = visualizations.len();
    let row_count = panel_count.div_ceil(COLUMNS);
    let content_width = (window.w() - CONTROL_PANEL_WIDTH).max(1.0);
    let content_center_x = window.left() + CONTROL_PANEL_WIDTH + content_width * 0.5;
    let cell_width = ((content_width - margin * 2.0 - gap * 2.0) / COLUMNS as f32).max(1.0);
    let cell_height = ((window.h() - margin * 2.0 - gap * row_count.saturating_sub(1) as f32)
        / row_count as f32)
        .max(1.0);
    let image_size = cell_width.min(cell_height - label_height - 12.0).max(1.0);
    for (index, visualization) in visualizations.iter().enumerate() {
        let row = index / COLUMNS;
        let column = index % COLUMNS;
        let items_in_row = (panel_count - row * COLUMNS).min(COLUMNS);
        let row_width =
            items_in_row as f32 * cell_width + items_in_row.saturating_sub(1) as f32 * gap;
        let x = content_center_x - row_width * 0.5
            + cell_width * 0.5
            + column as f32 * (cell_width + gap);
        let cell_top = window.top() - margin - row as f32 * (cell_height + gap);
        let cell_y = cell_top - cell_height * 0.5;
        let image_y = cell_top - label_height - 8.0 - image_size * 0.5;

        draw.rect()
            .x_y(x, cell_y)
            .w_h(cell_width, cell_height)
            .color(Color::srgb_u8(24, 28, 31));
        let label = match visualization {
            Visualization::Video => video_label,
            Visualization::Image(panel) => panel.label,
            Visualization::Laser { label, .. } => label,
        };
        draw.text(label)
            .x_y(x, cell_top - label_height * 0.5)
            .font_size(16)
            .color(Color::srgb_u8(210, 216, 220));

        match visualization {
            Visualization::Video => {
                if let Some(image) = &video.image {
                    let video_width = cell_width.min(image_size * 16.0 / 9.0);
                    draw.rect()
                        .x_y(x, image_y)
                        .w_h(video_width, video_width * 9.0 / 16.0)
                        .color(WHITE)
                        .texture(image);
                }
            }
            Visualization::Image(panel) => {
                draw.rect()
                    .x_y(x, image_y)
                    .w_h(image_size, image_size)
                    .color(WHITE)
                    .texture(&panel.image);
            }
            Visualization::Laser { path, .. } => {
                draw_laser_path(&draw, path, x, image_y, image_size * 0.5);
            }
        }
    }
}

fn draw_laser_path(
    draw: &Draw,
    laser_path: &path_generation::LaserPath,
    x: f32,
    y: f32,
    scale: f32,
) {
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
                    x + start.position[0] * scale,
                    y + start.position[1] * scale,
                ))
                .end(pt2(
                    x + end.position[0] * scale,
                    y + end.position[1] * scale,
                ))
                .weight(1.0)
                .color(Color::srgb(color[0], color[1], color[2]));
        }
    }

    for point in laser_path.laser_points() {
        draw.ellipse()
            .x_y(x + point.position[0] * scale, y + point.position[1] * scale)
            .radius(1.0)
            .color(Color::srgb(point.color[0], point.color[1], point.color[2]));
    }
}
