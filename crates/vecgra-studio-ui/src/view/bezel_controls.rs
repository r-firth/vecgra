use bezel_theme::Theme;
use bezel_ui::{
    control_bar::{self, Shape},
    widgets::{ButtonStyle, Buttons as _},
};
use gpui::{
    Context, InteractiveElement as _, IntoElement as _, ParentElement as _,
    StatefulInteractiveElement as _, Styled as _, div, px,
};

use super::{LayoutKind, StudioView};

impl StudioView {
    pub(super) fn render_bezel_graph_controls(&self, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = Theme::of(cx).clone();
        let ghost = ButtonStyle::Ghost;
        let layout_style = |kind| {
            if self.layout_kind == kind {
                ButtonStyle::Prominent
            } else {
                ghost
            }
        };

        let leading = vec![
            theme
                .button("−", ghost, Some("bezel-zoom-out".into()))
                .id("bezel-zoom-out")
                .on_click(cx.listener(|this, _, _, cx| this.zoom(0.82, cx)))
                .into_any_element(),
            theme
                .button("Fit", ghost, Some("bezel-fit-view".into()))
                .id("bezel-fit-view")
                .on_click(cx.listener(|this, _, _, cx| this.fit(cx)))
                .into_any_element(),
            theme
                .button("+", ghost, Some("bezel-zoom-in".into()))
                .debug_selector(|| "bezel-zoom-in".into())
                .id("bezel-zoom-in")
                .on_click(cx.listener(|this, _, _, cx| this.zoom(1.22, cx)))
                .into_any_element(),
        ];
        let trailing = vec![
            theme
                .button(
                    "Auto",
                    layout_style(LayoutKind::Auto),
                    Some("bezel-layout-auto".into()),
                )
                .id("bezel-layout-auto")
                .on_click(cx.listener(|this, _, window, cx| {
                    this.arrange(LayoutKind::Auto, window, cx);
                }))
                .into_any_element(),
            theme
                .button(
                    "Force",
                    layout_style(LayoutKind::Force),
                    Some("bezel-layout-force".into()),
                )
                .id("bezel-layout-force")
                .on_click(cx.listener(|this, _, window, cx| {
                    this.arrange(LayoutKind::Force, window, cx);
                }))
                .into_any_element(),
            theme
                .button(
                    "Orbit",
                    layout_style(LayoutKind::Orbit),
                    Some("bezel-layout-orbit".into()),
                )
                .id("bezel-layout-orbit")
                .on_click(cx.listener(|this, _, window, cx| {
                    this.arrange(LayoutKind::Orbit, window, cx);
                }))
                .into_any_element(),
            theme
                .button(
                    "Structure",
                    layout_style(LayoutKind::Structure),
                    Some("bezel-layout-structure".into()),
                )
                .id("bezel-layout-structure")
                .on_click(cx.listener(|this, _, window, cx| {
                    this.arrange(LayoutKind::Structure, window, cx);
                }))
                .into_any_element(),
        ];
        let centre = div()
            .font_family(theme.font_mono.clone())
            .text_size(px(12.0))
            .text_color(theme.text_muted)
            .child(format!("{:.0}%", self.camera.zoom * 100.0))
            .into_any_element();

        control_bar::control_bar(&theme, Shape::Pill, leading, Some(centre), trailing)
    }
}
