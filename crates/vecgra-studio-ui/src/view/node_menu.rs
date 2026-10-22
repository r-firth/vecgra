//! The menu a right click on a node opens: what the inspector and shortcuts
//! can do with a node, gathered where the pointer already is.

use super::*;
use bezel_theme::Theme as BezelTheme;
use bezel_ui::popover;

/// An open node menu.
pub(super) struct NodeMenu {
    /// The node it acts on, by ID so a reload cannot point it at another.
    pub(super) node: u64,
    /// Where it opened, in window coordinates.
    pub(super) position: Point<Pixels>,
    /// The node selected before the right click, which a path can be traced
    /// from.
    pub(super) previous: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NodeMenuItem {
    OpenContext,
    TracePathFrom(u64),
    SetPathOrigin,
    Release,
    CopyText,
    CopyId,
}

impl StudioView {
    /// A right click selects the node under the pointer and opens its menu;
    /// anywhere else closes the menu.
    pub(super) fn on_secondary_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus(&self.focus_handle, cx);
        let Some(index) = self.node_at(event.position) else {
            self.node_menu = None;
            cx.notify();
            return;
        };
        let Some(snapshot) = self.snapshot() else {
            return;
        };
        let node = snapshot.nodes.ids[index];
        let previous = self
            .selection
            .and_then(SceneSelection::node)
            .map(|selected| snapshot.nodes.ids[selected])
            .filter(|&previous| previous != node);
        self.status = format!("Selected {}", status_node_name(snapshot, index)).into();
        self.set_selection(Some(SceneSelection::Node(index)));
        self.hovered_node = None;
        self.node_menu = Some(NodeMenu {
            node,
            position: event.position,
            previous,
        });
        cx.stop_propagation();
        cx.notify();
    }

    /// The menu's rows for the open menu, in order; `None` marks a divider.
    pub(super) fn node_menu_items(&self) -> Vec<Option<(NodeMenuItem, String, &'static str)>> {
        let (Some(menu), Some(snapshot)) = (self.node_menu.as_ref(), self.snapshot()) else {
            return Vec::new();
        };
        let Some(index) = snapshot.node_index(menu.node) else {
            return Vec::new();
        };
        let mut items = vec![Some((
            NodeMenuItem::OpenContext,
            "Open connected context".to_string(),
            "↵",
        ))];
        // A chosen path origin wins over the previous selection, as it does
        // for a shift-click.
        let origin = match self.path_state {
            PathState::ChoosingEnd(draft) => Some(draft.start),
            PathState::Idle
            | PathState::Searching { .. }
            | PathState::Ready(_)
            | PathState::Failed { .. } => menu.previous,
        }
        .filter(|&origin| origin != menu.node);
        if let Some(origin) = origin
            && self.database_path.is_some()
        {
            items.push(Some((
                NodeMenuItem::TracePathFrom(origin),
                format!("Trace path from {}", self.status_node_name_by_id(origin)),
                "⇧ click",
            )));
        }
        if !matches!(self.path_state, PathState::ChoosingEnd(draft) if draft.start == menu.node) {
            items.push(Some((
                NodeMenuItem::SetPathOrigin,
                "Set as path origin".to_string(),
                "",
            )));
        }
        if self
            .workspace
            .as_ref()
            .is_some_and(|workspace| workspace.borrow().is_pinned(index))
        {
            items.push(Some((
                NodeMenuItem::Release,
                "Release pinned node".to_string(),
                "⇧⌘R",
            )));
        }
        items.push(None);
        items.push(Some((
            NodeMenuItem::CopyText,
            "Copy as text".to_string(),
            "⌘C",
        )));
        items.push(Some((NodeMenuItem::CopyId, "Copy ID".to_string(), "")));
        items
    }

    pub(super) fn choose_node_menu_item(
        &mut self,
        item: NodeMenuItem,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(menu) = self.node_menu.take() else {
            return;
        };
        let Some(index) = self
            .snapshot()
            .and_then(|snapshot| snapshot.node_index(menu.node))
        else {
            cx.notify();
            return;
        };
        match item {
            NodeMenuItem::OpenContext => self.focus_node_context(index, window, cx),
            NodeMenuItem::TracePathFrom(origin) => {
                if !matches!(self.path_state, PathState::ChoosingEnd(draft) if draft.start == origin)
                {
                    let Some(start) = self
                        .snapshot()
                        .and_then(|snapshot| snapshot.node_index(origin))
                    else {
                        cx.notify();
                        return;
                    };
                    self.choose_path_start(start, window, cx);
                }
                self.set_selection(Some(SceneSelection::Node(index)));
                self.trace_path_to_node(index, window, cx);
            }
            NodeMenuItem::SetPathOrigin => self.choose_path_start(index, window, cx),
            NodeMenuItem::Release => self.release_selected(cx),
            NodeMenuItem::CopyText => self.copy_selection(cx),
            NodeMenuItem::CopyId => {
                let id = format!("node:{}", menu.node);
                cx.write_to_clipboard(gpui::ClipboardItem::new_string(id.clone()));
                self.status = format!("Copied {id}").into();
            }
        }
        window.focus(&self.focus_handle, cx);
        cx.notify();
    }

    pub(super) fn render_node_menu(&self, cx: &Context<Self>) -> Option<gpui::AnyElement> {
        let menu = self.node_menu.as_ref()?;
        let items = self.node_menu_items();
        if items.is_empty() {
            return None;
        }
        let theme = BezelTheme::of(cx).clone();
        let rows = items.into_iter().map(|item| {
            let Some((item, label, hint)) = item else {
                return popover::divider().into_any_element();
            };
            let id = format!("node-menu:{item:?}");
            popover::menu_row(&theme, false, id.clone())
                .id(SharedString::from(id.clone()))
                .debug_selector(move || id.clone())
                .role(Role::MenuItem)
                .aria_label(label.clone())
                .justify_between()
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.choose_node_menu_item(item, window, cx);
                }))
                .child(label)
                .when(!hint.is_empty(), |this| {
                    this.child(popover::kbd_hint(&theme, hint))
                })
                .into_any_element()
        });
        let card = popover::popover_card(&theme)
            .id("node-menu")
            .role(Role::Menu)
            .aria_label("Node actions")
            .min_w(px(220.0))
            .max_w(px(340.0))
            .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                this.node_menu = None;
                cx.notify();
            }))
            .child(div().flex().flex_col().children(rows));
        Some(popover::menu_at(
            "node-menu",
            menu.position,
            card.into_any_element(),
            None,
        ))
    }
}
