use gpui::{App, Hsla, rgb};
use gpui_component::{Theme, ThemeMode};

#[derive(Clone, Copy)]
pub struct StudioPalette {
    pub graphite: Hsla,
    pub strata: Hsla,
    pub mist: Hsla,
    pub cobalt: Hsla,
    pub copper: Hsla,
    pub celadon: Hsla,
}

pub fn palette() -> StudioPalette {
    StudioPalette {
        graphite: rgb(0x0b1116).into(),
        strata: rgb(0x162129).into(),
        mist: rgb(0xd9e3e8).into(),
        cobalt: rgb(0x4d8dff).into(),
        copper: rgb(0xe49562).into(),
        celadon: rgb(0x65d1a5).into(),
    }
}

pub const RELATIONSHIP_COLOR_COUNT: usize = 7;

pub fn relationship_color_index(label: &str) -> usize {
    let hash = label.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    hash as usize % RELATIONSHIP_COLOR_COUNT
}

pub fn relationship_color(label: &str) -> Hsla {
    relationship_color_for_index(relationship_color_index(label))
}

pub fn relationship_color_for_index(index: usize) -> Hsla {
    let colors = palette();
    match index % RELATIONSHIP_COLOR_COUNT {
        0 => colors.copper,
        1 => colors.celadon,
        2 => colors.cobalt,
        3 => rgb(0x9b8cf2).into(),
        4 => rgb(0xd6c56b).into(),
        5 => rgb(0xd47c9c).into(),
        _ => rgb(0x60b8c9).into(),
    }
}

pub fn apply_studio_theme(cx: &mut App) {
    Theme::change(ThemeMode::Dark, None, cx);
    let colors = palette();
    let theme = Theme::global_mut(cx);
    theme.background = colors.graphite;
    theme.foreground = colors.mist;
    theme.border = rgb(0x26343d).into();
    theme.input = rgb(0x26343d).into();
    theme.ring = colors.cobalt;
    theme.primary = colors.cobalt;
    theme.primary_hover = rgb(0x6a9fff).into();
    theme.primary_active = rgb(0x3e79df).into();
    theme.primary_foreground = rgb(0xf7fbfd).into();
    theme.secondary = colors.strata;
    theme.secondary_hover = rgb(0x20303a).into();
    theme.secondary_active = rgb(0x293b47).into();
    theme.secondary_foreground = colors.mist;
    theme.accent = rgb(0x1e2d37).into();
    theme.accent_foreground = colors.mist;
    theme.muted = rgb(0x1b2831).into();
    theme.muted_foreground = rgb(0x8fa1ab).into();
    theme.popover = colors.strata;
    theme.popover_foreground = colors.mist;
    theme.sidebar = rgb(0x10181e).into();
    theme.sidebar_border = rgb(0x26343d).into();
    theme.sidebar_foreground = colors.mist;
    theme.sidebar_accent = rgb(0x1a2831).into();
    theme.sidebar_accent_foreground = colors.mist;
    theme.sidebar_primary = colors.cobalt;
    theme.sidebar_primary_foreground = rgb(0xf7fbfd).into();
    theme.title_bar = rgb(0x10181e).into();
    theme.title_bar_border = rgb(0x26343d).into();
    theme.status_bar = rgb(0x10181e).into();
    theme.status_bar_border = rgb(0x26343d).into();
    theme.selection = colors.cobalt.opacity(0.28);
    theme.success = colors.celadon;
    theme.warning = colors.copper;
    theme.link = colors.cobalt;
    theme.caret = colors.cobalt;
    theme.radius = gpui::px(5.0);
    theme.radius_lg = gpui::px(8.0);
    Theme::sync_base(cx);
}
