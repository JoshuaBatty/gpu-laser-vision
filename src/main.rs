mod edge_detection;
mod kernels;
mod laser;
mod path_generation;

use bevy::app::{App as BevyApp, Plugin, PostUpdate};
use bevy::window::{PrimaryWindow, Window};
use nannou::prelude::bevy_asset::{Assets, RenderAssetUsages};
use nannou::prelude::*;
use std::sync::{Arc, Mutex, MutexGuard};

const SOURCE_IMAGE: &str = "assets/test_tiles.png";
const COLUMNS: usize = 3;

type AppModel = Result<Model, String>;

struct Model {
    panels: [ImagePanel; 7],
    laser_path: path_generation::LaserPath,
    laser: laser::EtherDreamStream,
    video: Arc<Mutex<VideoBridgeState>>,
}

#[derive(Clone)]
struct ImagePanel {
    label: &'static str,
    image: Handle<Image>,
}

#[derive(Component, Clone)]
struct VideoBridge(Arc<Mutex<VideoBridgeState>>);

#[derive(Default)]
struct VideoBridgeState {
    image: Option<Handle<Image>>,
    processed: Option<ProcessedVideoFrame>,
    error: Option<String>,
    middle_seconds: Option<f64>,
    seek_requested: bool,
    window_sized: bool,
}

struct ProcessedVideoFrame {
    panels: [ImagePanel; 7],
    laser_path: path_generation::LaserPath,
}

struct VideoBridgePlugin;

impl Plugin for VideoBridgePlugin {
    fn build(&self, app: &mut BevyApp) {
        app.add_systems(PostUpdate, capture_middle_video_frame);
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
    let laser_path =
        path_generation::from_hysteresis(&images.connected_edges, &images.edge_colors);
    let laser = laser::EtherDreamStream::start(&laser_path);
    let video = Arc::new(Mutex::new(VideoBridgeState::default()));
    let video_asset = app.asset_server().load("big_buck_bunny_720p.mp4");
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
        laser_path,
        laser,
        video,
    })
}

fn capture_middle_video_frame(
    outputs: Query<(Entity, &VideoOutput, &VideoPlayer, &VideoBridge), Changed<VideoOutput>>,
    videos: Res<Assets<Video>>,
    mut assets: ResMut<Assets<Image>>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut commands: Commands,
) {
    for (entity, output, player, bridge) in &outputs {
        let should_process = {
            let mut video = lock_video(&bridge.0);
            video.image = Some(output.image.clone());
            if let Some(source) = videos.get(&player.video) {
                if !video.window_sized {
                    if let Ok(mut window) = windows.single_mut() {
                        window
                            .resolution
                            .set(source.size.x as f32, source.size.y as f32);
                        video.window_sized = true;
                    }
                }
                if video.middle_seconds.is_none() {
                    video.middle_seconds = source.duration_seconds.map(|duration| duration * 0.5);
                }
            }
            if !video.seek_requested {
                if let Some(middle) = video.middle_seconds {
                    commands.entity(entity).insert(SeekTo(middle));
                    video.seek_requested = true;
                }
                false
            } else {
                video.processed.is_none()
                    && video.error.is_none()
                    && video
                        .middle_seconds
                        .is_some_and(|middle| output.position_seconds >= middle)
            }
        };
        if !should_process {
            continue;
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

        match edge_detection::process_rgba(frame) {
            Ok(images) => {
                let laser_path = path_generation::from_hysteresis(
                    &images.connected_edges,
                    &images.edge_colors,
                );
                let panels = image_panels(images, |image| {
                    assets.add(Image::from_dynamic(
                        image,
                        true,
                        RenderAssetUsages::default(),
                    ))
                });
                lock_video(&bridge.0).processed = Some(ProcessedVideoFrame {
                    panels,
                    laser_path,
                });
                commands.entity(entity).insert(SeekTo(0.0));
            }
            Err(error) => {
                lock_video(&bridge.0).error = Some(format!("First-frame CUDA failed: {error:#}"));
            }
        }
    }
}

fn image_panels(
    images: edge_detection::EdgeDetectionImages,
    mut upload: impl FnMut(image::DynamicImage) -> Handle<Image>,
) -> [ImagePanel; 7] {
    [
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
            label: "Non-maximum suppression",
            image: upload(image::DynamicImage::ImageLuma8(images.thin_edges)),
        },
        ImagePanel {
            label: "Threshold classes",
            image: upload(image::DynamicImage::ImageLuma8(images.edge_classes)),
        },
        ImagePanel {
            label: "Hysteresis",
            image: upload(image::DynamicImage::ImageLuma8(images.connected_edges)),
        },
        ImagePanel {
            label: "GPU edge colours",
            image: upload(image::DynamicImage::ImageRgb8(images.edge_colors)),
        },
    ]
}

fn lock_video(video: &Mutex<VideoBridgeState>) -> MutexGuard<'_, VideoBridgeState> {
    video.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn view(app: &App, model: &AppModel, _window: Entity) {
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
    let video = lock_video(&model.video);
    let processed = video.processed.as_ref();
    let panels = processed.map_or(&model.panels, |frame| &frame.panels);
    let laser_path = processed.map_or(&model.laser_path, |frame| &frame.laser_path);

    let margin = 24.0;
    let gap = 18.0;
    let label_height = 30.0;
    let panel_count = panels.len() + 2;
    let row_count = panel_count.div_ceil(COLUMNS);
    let cell_width = ((window.w() - margin * 2.0 - gap * 2.0) / COLUMNS as f32).max(1.0);
    let cell_height = ((window.h()
        - margin * 2.0
        - gap * row_count.saturating_sub(1) as f32)
        / row_count as f32)
        .max(1.0);
    let image_size = cell_width
        .min(cell_height - label_height - 12.0)
        .max(1.0);
    let laser_path_label = format!(
        "Laser - {} lines / {} points - {}",
        laser_path.laser_lines().len(),
        laser_path.point_count(),
        model.laser.status()
    );
    let video_label = video.error.as_deref().unwrap_or("Big Buck Bunny - 720p");

    for index in 0..panel_count {
        let row = index / COLUMNS;
        let column = index % COLUMNS;
        let items_in_row = (panel_count - row * COLUMNS).min(COLUMNS);
        let row_width = items_in_row as f32 * cell_width
            + items_in_row.saturating_sub(1) as f32 * gap;
        let x = -row_width * 0.5 + cell_width * 0.5 + column as f32 * (cell_width + gap);
        let cell_top = window.top() - margin - row as f32 * (cell_height + gap);
        let cell_y = cell_top - cell_height * 0.5;
        let image_y = cell_top - label_height - 8.0 - image_size * 0.5;

        draw.rect()
            .x_y(x, cell_y)
            .w_h(cell_width, cell_height)
            .color(Color::srgb_u8(24, 28, 31));
        let label = if index == 0 {
            video_label
        } else {
            panels
                .get(index - 1)
                .map_or(laser_path_label.as_str(), |panel| panel.label)
        };
        draw.text(label)
            .x_y(x, cell_top - label_height * 0.5)
            .font_size(16)
            .color(Color::srgb_u8(210, 216, 220));

        if index == 0 {
            if let Some(image) = &video.image {
                let video_width = cell_width.min(image_size * 16.0 / 9.0);
                draw.rect()
                    .x_y(x, image_y)
                    .w_h(video_width, video_width * 9.0 / 16.0)
                    .color(WHITE)
                    .texture(image);
            }
        } else if let Some(panel) = panels.get(index - 1) {
            draw.rect()
                .x_y(x, image_y)
                .w_h(image_size, image_size)
                .color(WHITE)
                .texture(&panel.image);
        } else {
            let scale = image_size * 0.5;
            draw.x_y(x, image_y)
                .scale(scale)
                .path()
                .stroke()
                .weight(2.0 / scale)
                .color(Color::srgb_u8(40, 52, 49))
                .events(laser_path.lyon_path().iter());
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
                            image_y + start.position[1] * scale,
                        ))
                        .end(pt2(
                            x + end.position[0] * scale,
                            image_y + end.position[1] * scale,
                        ))
                        .weight(2.0)
                        .color(Color::srgb(color[0], color[1], color[2]));
                }
            }
            for point in laser_path.laser_points() {
                draw.ellipse()
                    .x_y(
                        x + point.position[0] * scale,
                        image_y + point.position[1] * scale,
                    )
                    .w_h(5.0, 5.0)
                    .color(Color::srgb(
                        point.color[0],
                        point.color[1],
                        point.color[2],
                    ));
            }
        }
    }
}
