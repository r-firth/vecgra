use bezel_theme::Theme;
use bezel_ui::{
    control_bar::{self, Shape},
    focus,
    widgets::{ButtonStyle, Buttons as _, Controls as _, Layout as _},
};
use gpui::prelude::FluentBuilder as _;
use gpui::{
    Context, InteractiveElement as _, IntoElement as _, ParentElement as _, Role,
    StatefulInteractiveElement as _, Styled as _, div, px,
};

use super::{LayoutKind, SceneSelection, SearchMode, StudioView};

impl StudioView {
    pub(super) fn render_bezel_search_modes(&self, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = Theme::of(cx).clone();

        theme
            .toggle_group()
            .self_center()
            .id("bezel-search-modes")
            .debug_selector(|| "bezel-search-modes".into())
            .role(Role::Group)
            .aria_label("Search mode")
            .child(
                focus::focusable(
                    &theme,
                    &self.bezel_search_focus[0],
                    theme.toggle_group_item("Text", self.search_mode == SearchMode::Text),
                )
                .id("bezel-search-text")
                .role(Role::Button)
                .aria_label("Use text search")
                .aria_selected(self.search_mode == SearchMode::Text)
                .on_click(cx.listener(|this, _, window, cx| {
                    this.set_search_mode(SearchMode::Text, window, cx);
                }))
                .on_action(cx.listener(|this, _: &focus::Activate, window, cx| {
                    this.set_search_mode(SearchMode::Text, window, cx);
                })),
            )
            .child(
                focus::focusable(
                    &theme,
                    &self.bezel_search_focus[1],
                    theme.toggle_group_item("Hybrid", self.search_mode == SearchMode::Hybrid),
                )
                .id("bezel-search-hybrid")
                .role(Role::Button)
                .aria_label("Use hybrid search")
                .aria_selected(self.search_mode == SearchMode::Hybrid)
                .on_click(cx.listener(|this, _, window, cx| {
                    this.set_search_mode(SearchMode::Hybrid, window, cx);
                }))
                .on_action(cx.listener(|this, _: &focus::Activate, window, cx| {
                    this.set_search_mode(SearchMode::Hybrid, window, cx);
                })),
            )
            .child(
                focus::focusable(
                    &theme,
                    &self.bezel_search_focus[2],
                    theme.toggle_group_item("Semantic", self.search_mode == SearchMode::Semantic),
                )
                .id("bezel-search-semantic")
                .debug_selector(|| "bezel-search-semantic".into())
                .role(Role::Button)
                .aria_label("Use semantic search")
                .aria_selected(self.search_mode == SearchMode::Semantic)
                .on_click(cx.listener(|this, _, window, cx| {
                    this.set_search_mode(SearchMode::Semantic, window, cx);
                }))
                .on_action(cx.listener(|this, _: &focus::Activate, window, cx| {
                    this.set_search_mode(SearchMode::Semantic, window, cx);
                })),
            )
            .into_any_element()
    }

    pub(super) fn render_bezel_zoom_controls(&self, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = Theme::of(cx).clone();

        div()
            .flex()
            .items_center()
            .gap(px(2.0))
            .child(
                focus::focusable(
                    &theme,
                    &self.bezel_zoom_focus[0],
                    theme.button("−", ButtonStyle::Ghost, Some("compact-zoom-out".into())),
                )
                .id("compact-zoom-out")
                .role(Role::Button)
                .aria_label("Zoom out")
                .on_click(cx.listener(|this, _, _, cx| this.zoom(0.82, cx)))
                .on_action(cx.listener(|this, _: &focus::Activate, _, cx| {
                    this.zoom(0.82, cx);
                })),
            )
            .child(
                focus::focusable(
                    &theme,
                    &self.bezel_zoom_focus[1],
                    theme.button("Fit", ButtonStyle::Ghost, Some("compact-fit-view".into())),
                )
                .id("compact-fit-view")
                .role(Role::Button)
                .aria_label("Fit graph to view")
                .on_click(cx.listener(|this, _, _, cx| this.fit(cx)))
                .on_action(cx.listener(|this, _: &focus::Activate, _, cx| this.fit(cx))),
            )
            .child(
                focus::focusable(
                    &theme,
                    &self.bezel_zoom_focus[2],
                    theme.button("+", ButtonStyle::Ghost, Some("compact-zoom-in".into())),
                )
                .id("compact-zoom-in")
                .role(Role::Button)
                .aria_label("Zoom in")
                .on_click(cx.listener(|this, _, _, cx| this.zoom(1.22, cx)))
                .on_action(cx.listener(|this, _: &focus::Activate, _, cx| {
                    this.zoom(1.22, cx);
                })),
            )
            .into_any_element()
    }

    pub(super) fn render_bezel_layout_controls(&self, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = Theme::of(cx).clone();
        let release_enabled = self.selected_node_is_pinned();
        let release_face = theme.button(
            "Release",
            ButtonStyle::Ghost,
            release_enabled.then(|| "compact-release-node".into()),
        );
        let release = if release_enabled {
            focus::focusable(&theme, &self.bezel_release_focus, release_face)
                .id("compact-release-node")
                .role(Role::Button)
                .aria_label("Release selected node from its pinned position")
                .on_click(cx.listener(|this, _, _, cx| this.release_selected(cx)))
                .on_action(cx.listener(|this, _: &focus::Activate, _, cx| {
                    this.release_selected(cx);
                }))
        } else {
            release_face
                .id("compact-release-node")
                .role(Role::Button)
                .aria_label("Release selected node from its pinned position")
                .opacity(0.38)
                .cursor_default()
        };

        div()
            .flex()
            .items_center()
            .gap(px(4.0))
            .child(
                theme
                    .toggle_group()
                    .self_center()
                    .id("bezel-layout-modes")
                    .debug_selector(|| "bezel-layout-modes".into())
                    .role(Role::Group)
                    .aria_label("Graph layout")
                    .children(
                        [
                            ("Auto", LayoutKind::Auto),
                            ("Force", LayoutKind::Force),
                            ("Structure", LayoutKind::Structure),
                            ("Orbit", LayoutKind::Orbit),
                        ]
                        .into_iter()
                        .enumerate()
                        .map(|(index, (label, kind))| {
                            focus::focusable(
                                &theme,
                                &self.bezel_layout_focus[index],
                                theme.toggle_group_item(label, self.layout_kind == kind),
                            )
                            .id(format!("compact-layout-{}", label.to_lowercase()))
                            .role(Role::Button)
                            .aria_label(format!("Use {label} graph layout"))
                            .aria_selected(self.layout_kind == kind)
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.arrange(kind, window, cx);
                            }))
                            .on_action(cx.listener(
                                move |this, _: &focus::Activate, window, cx| {
                                    this.arrange(kind, window, cx);
                                },
                            ))
                        }),
                    ),
            )
            .child(release)
            .into_any_element()
    }

    pub(super) fn render_bezel_sidebar_tabs(
        &self,
        active_view: Option<&'static str>,
        cx: &Context<Self>,
    ) -> gpui::AnyElement {
        let theme = Theme::of(cx).clone();

        theme
            .tab_bar()
            .id("bezel-sidebar-tabs")
            .debug_selector(|| "bezel-sidebar-tabs".into())
            .mx(px(6.0))
            .mt(px(10.0))
            .role(Role::Group)
            .aria_label("Graph view")
            .child(
                focus::focusable(
                    &theme,
                    &self.bezel_overview_focus,
                    theme.tab("Overview", active_view.is_none()),
                )
                .id("overview-navigation")
                .role(Role::Button)
                .aria_label("Return to graph overview")
                .aria_selected(active_view.is_none())
                .on_click(cx.listener(|this, _, window, cx| {
                    this.show_overview(window, cx);
                }))
                .on_action(cx.listener(|this, _: &focus::Activate, window, cx| {
                    this.show_overview(window, cx);
                })),
            )
            .when_some(active_view, |tabs, label| {
                tabs.child(theme.tab(label, true).cursor_default())
            })
            .into_any_element()
    }

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
            focus::focusable(
                &theme,
                &self.bezel_zoom_focus[0],
                theme.button("−", ghost, Some("bezel-zoom-out".into())),
            )
            .id("bezel-zoom-out")
            .on_click(cx.listener(|this, _, _, cx| this.zoom(0.82, cx)))
            .on_action(cx.listener(|this, _: &focus::Activate, _, cx| {
                this.zoom(0.82, cx);
            }))
            .into_any_element(),
            focus::focusable(
                &theme,
                &self.bezel_zoom_focus[1],
                theme.button("Fit", ghost, Some("bezel-fit-view".into())),
            )
            .id("bezel-fit-view")
            .on_click(cx.listener(|this, _, _, cx| this.fit(cx)))
            .on_action(cx.listener(|this, _: &focus::Activate, _, cx| this.fit(cx)))
            .into_any_element(),
            focus::focusable(
                &theme,
                &self.bezel_zoom_focus[2],
                theme.button("+", ghost, Some("bezel-zoom-in".into())),
            )
            .debug_selector(|| "bezel-zoom-in".into())
            .id("bezel-zoom-in")
            .on_click(cx.listener(|this, _, _, cx| this.zoom(1.22, cx)))
            .on_action(cx.listener(|this, _: &focus::Activate, _, cx| {
                this.zoom(1.22, cx);
            }))
            .into_any_element(),
        ];
        let layout_buttons = [
            ("Auto", LayoutKind::Auto, "bezel-layout-auto"),
            ("Force", LayoutKind::Force, "bezel-layout-force"),
            ("Structure", LayoutKind::Structure, "bezel-layout-structure"),
            ("Orbit", LayoutKind::Orbit, "bezel-layout-orbit"),
        ];
        let mut trailing = layout_buttons
            .into_iter()
            .enumerate()
            .map(|(index, (label, kind, id))| {
                focus::focusable(
                    &theme,
                    &self.bezel_layout_focus[index],
                    theme.button(label, layout_style(kind), Some(id.into())),
                )
                .id(id)
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.arrange(kind, window, cx);
                }))
                .on_action(cx.listener(move |this, _: &focus::Activate, window, cx| {
                    this.arrange(kind, window, cx);
                }))
                .into_any_element()
            })
            .collect::<Vec<_>>();
        let release_enabled = self.selected_node_is_pinned();
        let release_face = theme.button(
            "Release",
            ghost,
            release_enabled.then(|| "bezel-release-node".into()),
        );
        trailing.push(if release_enabled {
            focus::focusable(&theme, &self.bezel_release_focus, release_face)
                .id("bezel-release-node")
                .role(Role::Button)
                .aria_label("Release selected node from its pinned position")
                .on_click(cx.listener(|this, _, _, cx| this.release_selected(cx)))
                .on_action(cx.listener(|this, _: &focus::Activate, _, cx| {
                    this.release_selected(cx);
                }))
                .into_any_element()
        } else {
            release_face
                .id("bezel-release-node")
                .role(Role::Button)
                .aria_label("Release selected node from its pinned position")
                .opacity(0.38)
                .cursor_default()
                .into_any_element()
        });
        let centre = div()
            .font_family(theme.font_mono.clone())
            .text_size(px(12.0))
            .text_color(theme.text_muted)
            .child(format!("{:.0}%", self.camera.zoom * 100.0))
            .into_any_element();

        control_bar::control_bar(&theme, Shape::Pill, leading, Some(centre), trailing)
    }

    fn selected_node_is_pinned(&self) -> bool {
        self.selection
            .and_then(SceneSelection::node)
            .is_some_and(|index| {
                self.workspace
                    .as_ref()
                    .is_some_and(|workspace| workspace.borrow().is_pinned(index))
            })
    }
}
