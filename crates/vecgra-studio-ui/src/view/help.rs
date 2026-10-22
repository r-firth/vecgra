use super::*;
use bezel_theme::Theme as BezelTheme;
use bezel_ui::popover;

/// Keyboard shortcuts, grouped as the help overlay lists them. Keep these in
/// step with the bindings in `vecgra-studio`'s `main.rs`.
const SHORTCUT_GROUPS: &[(&str, &[(&str, &str)])] = &[
    (
        "Navigate",
        &[
            ("⌘K", "Search or run a command"),
            ("↑  ↓", "Step through results, or earlier commands"),
            ("⌘0", "Fit the graph"),
            ("⌘=  ⌘-", "Zoom in and out"),
            ("Enter", "Lay out the selection's context"),
            ("Esc", "Leave a search, path, or lens"),
        ],
    ),
    (
        "Selection",
        &[
            ("⌘[  ⌘]", "Back and forward"),
            ("⌘C", "Copy the selection as text"),
            ("⇧ click", "Trace a path from the selection"),
            ("Right click", "A node's actions"),
            ("Tab  Enter", "Walk the inspector from the keyboard"),
            ("⇧⌘R", "Release a pinned node"),
        ],
    ),
    (
        "Layout",
        &[
            ("⌘1  ⌘2", "Auto and force"),
            ("⌘3  ⌘4", "Orbit and structure"),
        ],
    ),
    (
        "Database",
        &[
            ("⌘O", "Open a database"),
            ("⇧⌘E", "Export the view as SVG"),
            ("⌘R", "Reload from disk"),
            ("⇧⌘L", "Turn live refresh on or off"),
            ("⌘/", "Show or hide this help"),
        ],
    ),
];

/// Commands the search box understands. The first field is the syntax shown;
/// the second is what clicking the row puts in the search box.
const COMMANDS: &[(&str, &str, &str)] = &[
    ("node <id>  ·  edge <id>", "node ", "Select by ID"),
    ("center <id>", "center ", "Move the camera to a node"),
    ("focus <id>", "focus ", "Lay out a node's context"),
    (
        "path <from> <to> [out|in] [label] [hops]",
        "path ",
        "Trace an evidence path",
    ),
    (
        "facet node|relationship <label>",
        "facet node ",
        "Lens on one label or type",
    ),
    (
        "text | semantic | hybrid <query>",
        "semantic ",
        "Search in a given mode",
    ),
    (
        "layout auto|force|orbit|structure",
        "layout ",
        "Rearrange the graph",
    ),
    ("zoom <factor>  ·  fit", "zoom ", "Set the zoom"),
    ("open <path>", "open ", "Open a .vg database"),
    (
        "export <path.svg>",
        "export ",
        "Save the view as an SVG image",
    ),
    (
        "reload  ·  live on|off",
        "live ",
        "Reread or watch the file",
    ),
    (
        "back  ·  forward  ·  clear",
        "back",
        "Walk the selection history",
    ),
];

impl StudioView {
    pub(super) fn set_help_visible(
        &mut self,
        visible: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.help_visible == visible {
            return;
        }
        self.help_visible = visible;
        if visible {
            // Escape reaches the view only from its own focus; the search box
            // keeps Escape for itself.
            window.focus(&self.focus_handle, cx);
        }
        cx.notify();
    }

    /// Closes the help and starts a command in the search box.
    pub(super) fn begin_command(
        &mut self,
        prefix: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.help_visible = false;
        self.query_input.update(cx, |input, cx| {
            input.set_value(prefix, window, cx);
        });
        self.focus_search(window, cx);
        cx.notify();
    }

    pub(super) fn render_help(&self, window: &Window, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = BezelTheme::of(cx).clone();
        let mono = cx.theme().mono_font_family.clone();
        let shortcuts = v_flex()
            .id("help-shortcuts")
            .w(px(320.0))
            .flex_shrink_0()
            .gap_1()
            .children(SHORTCUT_GROUPS.iter().map(|(group, entries)| {
                v_flex()
                    .child(popover::menu_heading(&theme, *group).px_0())
                    .children(entries.iter().map(|(keys, description)| {
                        h_flex()
                            .h(px(24.0))
                            .gap_3()
                            .items_center()
                            .child(
                                h_flex().w(px(72.0)).flex_shrink_0().gap_1().children(
                                    keys.split("  ")
                                        .map(|key| popover::kbd_hint(&theme, key.to_string())),
                                ),
                            )
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .truncate()
                                    .text_sm()
                                    .text_color(cx.theme().foreground)
                                    .child(*description),
                            )
                    }))
            }));
        let commands = v_flex()
            .id("help-commands")
            .flex_1()
            .min_w_0()
            .child(popover::menu_heading(&theme, "Commands").px_0())
            .child(
                div()
                    .pb_2()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("Type in the search box. Anything else searches the graph."),
            )
            .children(COMMANDS.iter().enumerate().map(
                |(position, (syntax, prefix, description))| {
                    let prefix = *prefix;
                    v_flex()
                        .id(("help-command", position))
                        .debug_selector(move || format!("help-command:{prefix}"))
                        .role(Role::Button)
                        .aria_label(*description)
                        .px_2()
                        .py(px(3.0))
                        .mx(px(-8.0))
                        .rounded(px(6.0))
                        .cursor_pointer()
                        .hover(|this| this.bg(cx.theme().secondary_hover))
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.begin_command(prefix, window, cx);
                        }))
                        .child(
                            div()
                                .font_family(mono.clone())
                                .text_xs()
                                .text_color(palette().celadon)
                                .child(*syntax),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .child(*description),
                        )
                },
            ));
        let card = popover::dialog_card(&theme)
            .id("help-card")
            .debug_selector(|| "help-overlay".into())
            .role(Role::Dialog)
            .aria_label("Keyboard shortcuts and commands")
            .w(px(720.0))
            .max_h(window.viewport_size().height - px(96.0))
            .overflow_y_scroll()
            .child(
                h_flex()
                    .items_center()
                    .justify_between()
                    .pb_3()
                    .child(popover::dialog_title(&theme, "Shortcuts and commands"))
                    .child(popover::kbd_hint(&theme, "esc")),
            )
            .child(
                h_flex()
                    .items_start()
                    .gap_6()
                    .child(shortcuts)
                    .child(commands),
            );
        let entity = cx.entity();
        popover::modal(
            "help-modal",
            window.viewport_size(),
            card.into_any_element(),
            move |_, window, cx| {
                entity.update(cx, |this, cx| this.set_help_visible(false, window, cx));
            },
        )
    }
}
