//! GPUI views and components for Vecgra Studio.

use bezel_theme::{Appearance as BezelAppearance, Theme as BezelTheme};
use gpui::App;

mod graph_canvas;
mod theme;
mod view;

pub use theme::apply_studio_theme;
pub use view::{
    ActivateSelection, ArrangeAuto, ArrangeForce, ArrangeOrbit, ArrangeStructure, ClearSelection,
    FitView, FocusSearch, NextSearchResult, PreviousSearchResult, ReleaseSelected, StudioView,
    ZoomIn, ZoomOut,
};

/// Install Bezel's environment theme and bundled fonts for the Studio chrome.
pub fn init_bezel(cx: &mut App) {
    ensure_bezel_theme(cx);
    if let Err(error) = bezel_ui::register_fonts(cx) {
        eprintln!("Vecgra Studio could not register Bezel fonts: {error}");
    }
}

pub(crate) fn ensure_bezel_theme(cx: &mut App) {
    if !cx.has_global::<BezelTheme>() {
        BezelTheme::install(BezelAppearance::Dark, cx);
    }
}
