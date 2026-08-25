mod edge_detection;
mod kernels;

use nannou::prelude::*;
use nannou::prelude::bevy_asset::RenderAssetUsages;

const WINDOW_WIDTH: u32 = 1400;
const WINDOW_HEIGHT: u32 = 900;
const COLUMNS: usize = 3;

type AppModel = Result<Model, String>;

struct Model {
    panels: [ImagePanel; 5],
}

struct ImagePanel {
    label: &'static str,
    image: Handle<Image>,
}

fn main() {
    nannou::app(model).view(view).run();
}

fn model(app: &App) -> AppModel {
    app.new_window::<AppModel>()
        .size(WINDOW_WIDTH, WINDOW_HEIGHT)
        .title("GPU Laser Vision")
        .build();

    let images = edge_detection::process("assets/test_tiles.png").map_err(|error| {
        let error = format!("{error:#}");
        eprintln!("Edge detection failed: {error}");
        error
    })?;

    let upload = |image| {
        let image = Image::from_dynamic(
            image::DynamicImage::ImageLuma8(image),
            true,
            RenderAssetUsages::default(),
        );
        app.asset_server().add(image)
    };

    Ok(Model {
        panels: [
            ImagePanel {
                label: "Grayscale",
                image: upload(images.grayscale),
            },
            ImagePanel {
                label: "Scharr magnitude",
                image: upload(images.edges),
            },
            ImagePanel {
                label: "Non-maximum suppression",
                image: upload(images.thin_edges),
            },
            ImagePanel {
                label: "Threshold classes",
                image: upload(images.edge_classes),
            },
            ImagePanel {
                label: "Hysteresis",
                image: upload(images.connected_edges),
            },
        ],
    })
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

    let margin = 24.0;
    let gap = 18.0;
    let label_height = 30.0;
    let cell_width = ((window.w() - margin * 2.0 - gap * 2.0) / COLUMNS as f32).max(1.0);
    let cell_height = ((window.h() - margin * 2.0 - gap) / 2.0).max(1.0);
    let image_size = cell_width
        .min(cell_height - label_height - 12.0)
        .max(1.0);

    for (index, panel) in model.panels.iter().enumerate() {
        let row = index / COLUMNS;
        let column = index % COLUMNS;
        let items_in_row = (model.panels.len() - row * COLUMNS).min(COLUMNS);
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
        draw.text(panel.label)
            .x_y(x, cell_top - label_height * 0.5)
            .font_size(16)
            .color(Color::srgb_u8(210, 216, 220));
        draw.rect()
            .x_y(x, image_y)
            .w_h(image_size, image_size)
            .color(WHITE)
            .texture(&panel.image);
    }
}
