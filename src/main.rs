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
const DEBUG_UPLOAD_INTERVAL: u64 = 3;
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
    reported_first_frame: bool,
    debug_upload_counter: u64,
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
    let cuda_laser_path = path_generation::from_edge_mask(&images.laser_edges, &images.edge_colors);
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
            if !video.window_sized {
                if let Ok(mut window) = windows.single_mut() {
                    window
                        .resolution
                        .set(output.size.x as f32, output.size.y as f32);
                    video.window_sized = true;
                }
            }
        }

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

        let frame_started = Instant::now();
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
            .as_ref()
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
        match detector.process(&frame, thresholds.min, thresholds.max) {
            Ok(images) => {
                let cuda_laser_path =
                    path_generation::from_edge_mask(&images.laser_edges, &images.edge_colors);
                let yolo_laser_path = path_generation::from_edge_mask(
                    &yolo_frame.contour,
                    &yolo_frame.colored_contour,
                );
                let mut video = lock_video(&bridge.0);
                let cuda_point_count = cuda_laser_path.point_count();
                let yolo_point_count = yolo_laser_path.point_count();
                let upload_debug = if video.processed.is_some() {
                    video.debug_upload_counter += 1;
                    let upload = video.debug_upload_counter >= DEBUG_UPLOAD_INTERVAL;
                    if upload {
                        video.debug_upload_counter = 0;
                    }
                    upload
                } else {
                    true
                };
                if let Some(processed) = &mut video.processed {
                    update_processed_frame(
                        processed,
                        images,
                        yolo_frame,
                        cuda_laser_path,
                        yolo_laser_path,
                        upload_debug,
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
                    video.fps_window_started = Some(now);
                    video.processed_frames = 0;
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
    upload_debug: bool,
    assets: &mut Assets<Image>,
) {
    processed.cuda_laser_path = cuda_laser_path;
    processed.yolo_laser_path = yolo_laser_path;
    if !upload_debug {
        return;
    }

    let images = [
        image::DynamicImage::ImageRgb8(images.edge_colors),
        image::DynamicImage::ImageRgb8(yolo_frame.colored_contour),
        image::DynamicImage::ImageLuma8(yolo_frame.person_mask),
        image::DynamicImage::ImageLuma8(images.grayscale),
        image::DynamicImage::ImageLuma8(images.edges),
        image::DynamicImage::ImageLuma8(images.laser_edges),
    ];

    for (panel, image) in processed.panels.iter_mut().zip(images) {
        let image = Image::from_dynamic(image, true, RenderAssetUsages::default());
        if let Some(mut current) = assets.get_mut(&panel.image) {
            current.data = image.data;
        } else {
            panel.image = assets.add(image);
        }
    }
}

fn image_panels(
    images: edge_detection::EdgeDetectionImages,
    mut upload: impl FnMut(image::DynamicImage) -> Handle<Image>,
) -> Vec<ImagePanel> {
    vec![
        ImagePanel {
            label: "Original",
            image: upload(image::DynamicImage::ImageRgba8(images.original)),
        },
        ImagePanel {
            label: "Grayscale",
            image: upload(image::DynamicImage::ImageLuma8(images.grayscale)),
        },
        ImagePanel {
            label: "Scharr magnitude",
            image: upload(image::DynamicImage::ImageLuma8(images.edges)),
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
            image: upload(image::DynamicImage::ImageLuma8(images.grayscale)),
        },
        ImagePanel {
            label: "Scharr magnitude",
            image: upload(image::DynamicImage::ImageLuma8(images.edges)),
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
