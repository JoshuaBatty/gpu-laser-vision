use nannou::prelude::*;

pub const HEADER_HEIGHT: f32 = 68.0;
pub const SIDEBAR_WIDTH: f32 = 244.0;

const PAGE_MARGIN: f32 = 16.0;
const CARD_GAP: f32 = 12.0;
const CARD_HEADER_HEIGHT: f32 = 38.0;
const CARD_PADDING: f32 = 9.0;
const MEDIA_ASPECT_RATIO: f32 = 16.0 / 9.0;

#[derive(Clone, Copy)]
pub enum Accent {
    Neutral,
    Cuda,
    Yolo,
    Diagnostic,
    Error,
}

impl Accent {
    pub fn draw_color(self) -> Color {
        match self {
            Self::Neutral => Color::srgb_u8(120, 132, 145),
            Self::Cuda => Color::srgb_u8(255, 176, 76),
            Self::Yolo => Color::srgb_u8(52, 211, 153),
            Self::Diagnostic => Color::srgb_u8(91, 105, 117),
            Self::Error => Color::srgb_u8(248, 113, 113),
        }
    }

    pub fn ui_color(self) -> egui::Color32 {
        match self {
            Self::Neutral => egui::Color32::from_rgb(120, 132, 145),
            Self::Cuda => egui::Color32::from_rgb(255, 176, 76),
            Self::Yolo => egui::Color32::from_rgb(52, 211, 153),
            Self::Diagnostic => egui::Color32::from_rgb(91, 105, 117),
            Self::Error => egui::Color32::from_rgb(248, 113, 113),
        }
    }
}

#[derive(Clone, Copy)]
pub struct CardRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl CardRect {
    pub fn top(self) -> f32 {
        self.y + self.height * 0.5
    }

    pub fn media_rect(self) -> Self {
        let available_width = (self.width - CARD_PADDING * 2.0).max(1.0);
        let available_height = (self.height - CARD_HEADER_HEIGHT - CARD_PADDING * 2.0).max(1.0);
        let (width, height) = if available_width / available_height > MEDIA_ASPECT_RATIO {
            (available_height * MEDIA_ASPECT_RATIO, available_height)
        } else {
            (available_width, available_width / MEDIA_ASPECT_RATIO)
        };
        let content_top = self.top() - CARD_HEADER_HEIGHT - CARD_PADDING;

        Self {
            x: self.x,
            y: content_top - available_height * 0.5,
            width,
            height,
        }
    }
}

pub struct DashboardLayout {
    pub hero: [CardRect; 3],
    pub outputs: [CardRect; 2],
    pub diagnostics: [CardRect; 4],
}

impl DashboardLayout {
    pub fn new(window: Rect) -> Self {
        let content_left = window.left() + SIDEBAR_WIDTH + PAGE_MARGIN;
        let content_right = window.right() - PAGE_MARGIN;
        let content_top = window.top() - HEADER_HEIGHT - PAGE_MARGIN;
        let content_bottom = window.bottom() + PAGE_MARGIN;
        let content_width = (content_right - content_left).max(1.0);
        let available_height = (content_top - content_bottom - CARD_GAP * 2.0).max(3.0);

        let hero_height = available_height * 0.36;
        let output_height = available_height * 0.34;
        let diagnostic_height = available_height - hero_height - output_height;

        let hero_top = content_top;
        let output_top = hero_top - hero_height - CARD_GAP;
        let diagnostic_top = output_top - output_height - CARD_GAP;

        Self {
            hero: row::<3>(content_left, hero_top, content_width, hero_height),
            outputs: row::<2>(content_left, output_top, content_width, output_height),
            diagnostics: row::<4>(
                content_left,
                diagnostic_top,
                content_width,
                diagnostic_height,
            ),
        }
    }
}

fn row<const COUNT: usize>(
    left: f32,
    top: f32,
    available_width: f32,
    height: f32,
) -> [CardRect; COUNT] {
    let total_gap = CARD_GAP * COUNT.saturating_sub(1) as f32;
    let width = ((available_width - total_gap) / COUNT as f32).max(1.0);
    std::array::from_fn(|index| CardRect {
        x: left + width * 0.5 + index as f32 * (width + CARD_GAP),
        y: top - height * 0.5,
        width,
        height,
    })
}

pub fn background() -> Color {
    Color::srgb_u8(8, 11, 14)
}

pub fn configure_egui(context: &egui::Context) {
    let mut style = (*context.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    style.spacing.slider_width = 132.0;
    style.visuals.panel_fill = egui::Color32::from_rgb(14, 18, 22);
    style.visuals.window_fill = egui::Color32::from_rgb(14, 18, 22);
    style.visuals.extreme_bg_color = egui::Color32::from_rgb(8, 11, 14);
    style.visuals.faint_bg_color = egui::Color32::from_rgb(22, 28, 33);
    style.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(27, 34, 40);
    style.visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(27, 34, 40);
    style.visuals.widgets.inactive.fg_stroke.color = egui::Color32::from_rgb(183, 192, 199);
    style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(37, 46, 53);
    style.visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(37, 46, 53);
    style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(45, 56, 64);
    style.visuals.selection.bg_fill = egui::Color32::from_rgb(155, 101, 41);
    style.visuals.override_text_color = Some(egui::Color32::from_rgb(220, 226, 230));
    context.set_global_style(style);
}

pub fn header_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(11, 15, 18))
        .inner_margin(egui::Margin::symmetric(18, 10))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(34, 42, 48)))
}

pub fn sidebar_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(14, 18, 22))
        .inner_margin(egui::Margin::symmetric(16, 18))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(34, 42, 48)))
}

pub fn section_label(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(10.0)
            .strong()
            .color(egui::Color32::from_rgb(118, 132, 143)),
    );
}

pub fn metric_row(ui: &mut egui::Ui, label: &str, value: impl Into<egui::WidgetText>) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .size(12.0)
                .color(egui::Color32::from_rgb(148, 160, 169)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(value);
        });
    });
}

pub fn status_chip(ui: &mut egui::Ui, label: &str, value: &str, accent: Accent) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(20, 26, 31))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(43, 53, 60)))
        .corner_radius(5)
        .inner_margin(egui::Margin::symmetric(9, 5))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(label)
                        .size(10.0)
                        .strong()
                        .color(egui::Color32::from_rgb(117, 130, 140)),
                );
                ui.label(
                    egui::RichText::new(value)
                        .size(11.0)
                        .strong()
                        .color(accent.ui_color()),
                );
            });
        });
}

pub fn draw_card_shell(
    draw: &Draw,
    card: CardRect,
    eyebrow: &str,
    title: &str,
    meta: &str,
    accent: Accent,
) -> CardRect {
    draw.rect()
        .x_y(card.x, card.y)
        .w_h(card.width, card.height)
        .color(Color::srgb_u8(17, 22, 26))
        .stroke(Color::srgb_u8(38, 47, 54))
        .stroke_weight(1.0);
    draw.rect()
        .x_y(card.x, card.top() - 1.5)
        .w_h(card.width, 3.0)
        .color(accent.draw_color());

    let text_width = (card.width - CARD_PADDING * 2.0).max(1.0);
    draw.text(eyebrow)
        .x_y(card.x, card.top() - 11.0)
        .w_h(text_width, 12.0)
        .left_justify()
        .font_size(9)
        .color(Color::srgb_u8(106, 120, 131));
    draw.text(title)
        .x_y(card.x, card.top() - 26.0)
        .w_h(text_width, 17.0)
        .left_justify()
        .font_size(13)
        .color(Color::srgb_u8(220, 226, 230));
    draw.text(meta)
        .x_y(card.x, card.top() - 11.0)
        .w_h(text_width, 12.0)
        .right_justify()
        .font_size(9)
        .color(accent.draw_color());

    let media = card.media_rect();
    draw.rect()
        .x_y(media.x, media.y)
        .w_h(media.width, media.height)
        .color(Color::srgb_u8(2, 4, 5));
    media
}

pub fn draw_texture(draw: &Draw, image: &Handle<Image>, media: CardRect) {
    draw.rect()
        .x_y(media.x, media.y)
        .w_h(media.width, media.height)
        .color(WHITE)
        .texture(image);
}

pub fn draw_empty_state(draw: &Draw, media: CardRect, message: &str) {
    draw.text(message)
        .x_y(media.x, media.y)
        .w_h(media.width - 16.0, media.height)
        .font_size(11)
        .color(Color::srgb_u8(87, 101, 111));
}
