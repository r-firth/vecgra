//! The menu a right click on an overview label or type opens: its lens,
//! hiding it, and showing only it, so none of these needs a node to start
//! from or a command to remember.

use super::*;
use bezel_theme::Theme as BezelTheme;
use bezel_ui::popover;

/// An open overview row menu.
pub(super) struct FacetMenu {
    pub(super) facet: FacetLens,
    /// Where it opened, in window coordinates.
    pub(super) position: Point<Pixels>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FacetMenuItem {
    Lens,
    Hide,
    ShowOnly,
}

impl StudioView {
    pub(super) fn open_facet_menu(
        &mut self,
        facet: FacetLens,
        position: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus(&self.focus_handle, cx);
        self.node_menu = None;
        self.facet_menu = Some(FacetMenu { facet, position });
        cx.notify();
    }

    /// The menu's rows for the open menu, in order.
    pub(super) fn facet_menu_items(&self) -> Vec<(FacetMenuItem, String)> {
        let Some(menu) = self.facet_menu.as_ref() else {
            return Vec::new();
        };
        let (counts, elements, element) = match &menu.facet {
            FacetLens::NodeLabel(_) => (&self.node_label_counts, "nodes", "node"),
            FacetLens::Relationship(_) => {
                (&self.relationship_counts, "relationships", "relationship")
            }
        };
        let label = menu.facet.label();
        let Some(count) = counts
            .iter()
            .find(|(other, _)| other == label)
            .map(|(_, count)| *count)
        else {
            return Vec::new();
        };
        let mut items = vec![
            (
                FacetMenuItem::Lens,
                if self.active_facet.as_ref() == Some(&menu.facet) {
                    "Clear lens".to_string()
                } else {
                    format!("Lens on {label}")
                },
            ),
            (
                FacetMenuItem::Hide,
                if count == 1 {
                    format!("Hide the {label} {element}")
                } else {
                    format!(
                        "Hide all {} {label} {elements}",
                        panels::format_count(count)
                    )
                },
            ),
        ];
        if counts.len() > 1 {
            items.push((
                FacetMenuItem::ShowOnly,
                format!("Show only {label} {elements}"),
            ));
        }
        items
    }

    pub(super) fn choose_facet_menu_item(
        &mut self,
        item: FacetMenuItem,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(menu) = self.facet_menu.take() else {
            return;
        };
        let label = menu.facet.label().clone();
        let relationship = matches!(menu.facet, FacetLens::Relationship(_));
        match item {
            FacetMenuItem::Lens => self.activate_facet(menu.facet, window, cx),
            FacetMenuItem::Hide if relationship => self.hide_relationship(&label, cx),
            FacetMenuItem::Hide => self.hide_label(&label, cx),
            FacetMenuItem::ShowOnly => self.show_only(&label, relationship, cx),
        }
        window.focus(&self.focus_handle, cx);
        cx.notify();
    }

    pub(super) fn render_facet_menu(&self, cx: &Context<Self>) -> Option<gpui::AnyElement> {
        let menu = self.facet_menu.as_ref()?;
        let items = self.facet_menu_items();
        if items.is_empty() {
            return None;
        }
        let theme = BezelTheme::of(cx).clone();
        let rows = items.into_iter().map(|(item, label)| {
            let id = format!("facet-menu:{item:?}");
            popover::menu_row(&theme, false, id.clone())
                .id(SharedString::from(id.clone()))
                .debug_selector(move || id.clone())
                .role(Role::MenuItem)
                .aria_label(label.clone())
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.choose_facet_menu_item(item, window, cx);
                }))
                .child(label)
                .into_any_element()
        });
        let card = popover::popover_card(&theme)
            .id("facet-menu")
            .role(Role::Menu)
            .aria_label(format!("{} actions", menu.facet.label()))
            .min_w(px(220.0))
            .max_w(px(340.0))
            .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                this.facet_menu = None;
                cx.notify();
            }))
            .child(div().flex().flex_col().children(rows));
        Some(popover::menu_at(
            "facet-menu",
            menu.position,
            card.into_any_element(),
            None,
        ))
    }
}
