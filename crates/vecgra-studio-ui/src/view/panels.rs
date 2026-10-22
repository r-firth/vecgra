use super::*;
use bezel_theme::Theme as BezelTheme;
use bezel_ui::{icons, popover, tooltip::Tooltip, widgets::Content as _};

#[cfg(target_os = "macos")]
const TITLE_BAR_PLATFORM_INSET: f32 = 80.0;
#[cfg(not(target_os = "macos"))]
const TITLE_BAR_PLATFORM_INSET: f32 = 12.0;
const TOOLBAR_GAP: f32 = 8.0;
const OVERVIEW_SIDEBAR_WIDTH: f32 = 218.0;
const INSPECTOR_PROPERTY_LIMIT: usize = 24;
const INSPECTOR_CONNECTIONS_PER_GROUP: usize = 6;
/// How many of a group's connections the inspector lists once expanded.
pub(super) const INSPECTOR_CONNECTIONS_EXPANDED: usize = 100;
const INSPECTOR_CONNECTION_GROUPS: usize = 8;
const DETAIL_SIDEBAR_WIDTH: f32 = 306.0;

impl StudioView {
    pub(super) fn render_brand(
        &self,
        database_name: SharedString,
        width: f32,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        h_flex()
            .w(px(width))
            .flex_shrink_0()
            .gap_2()
            .items_center()
            .child(
                div()
                    .size(px(24.0))
                    .flex_none()
                    .rounded(px(6.0))
                    .overflow_hidden()
                    .child(
                        gpui::img(vecgra_logo_image())
                            .size(px(38.0))
                            .ml(px(-7.0))
                            .mt(px(-7.0)),
                    ),
            )
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .line_height(px(15.0))
                    .child(
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child("Vecgra"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .truncate()
                            .child(database_name),
                    ),
            )
    }

    pub(super) fn render_top_bar(&self, compact: bool, cx: &Context<Self>) -> gpui::AnyElement {
        let database_name: SharedString = match &self.state {
            LoadState::Ready(snapshot) => snapshot.database_name.to_string().into(),
            LoadState::Loading { name } => name.clone(),
            LoadState::Failed(_) => "No database".into(),
        };
        let brand_width =
            (self.left_panel_width() - TITLE_BAR_PLATFORM_INSET - TOOLBAR_GAP).max(120.0);
        if compact {
            return TitleBar::new()
                .pl_0()
                .h(px(88.0))
                .child(
                    v_flex()
                        .size_full()
                        .child(
                            h_flex()
                                .debug_selector(|| "compact-toolbar-primary".into())
                                .h(px(48.0))
                                .pl(px(TITLE_BAR_PLATFORM_INSET))
                                .pr_3()
                                .gap_2()
                                .items_center()
                                .child(self.render_brand(database_name, brand_width, cx))
                                .child(
                                    div()
                                        .debug_selector(|| "compact-search-field".into())
                                        .flex_1()
                                        .min_w(px(180.0))
                                        .child(self.render_query_input(cx)),
                                )
                                .child(self.render_bezel_zoom_controls(cx)),
                        )
                        .child(
                            h_flex()
                                .debug_selector(|| "compact-toolbar-secondary".into())
                                .h(px(40.0))
                                .px_3()
                                .gap_2()
                                .items_center()
                                .border_t_1()
                                .border_color(cx.theme().title_bar_border)
                                .child(self.render_bezel_search_modes(cx))
                                .child(div().flex_1())
                                .child(self.render_bezel_layout_controls(cx)),
                        ),
                )
                .into_any_element();
        }
        TitleBar::new()
            .pl_0()
            .h(px(52.0))
            .child(
                h_flex()
                    .debug_selector(|| "wide-toolbar".into())
                    .size_full()
                    .pl(px(TITLE_BAR_PLATFORM_INSET))
                    .pr_3()
                    .gap_2()
                    .items_center()
                    .child(self.render_brand(database_name, brand_width, cx))
                    .child(
                        div()
                            .debug_selector(|| "wide-search-field".into())
                            .flex_1()
                            .min_w(px(180.0))
                            .max_w(px(620.0))
                            .child(self.render_query_input(cx)),
                    )
                    .child(self.render_bezel_search_modes(cx))
                    .child(div().flex_1()),
            )
            .into_any_element()
    }

    /// The command box, ending in a hint that says what Enter will do: run
    /// the command typed, search, or go to the selected result of the search
    /// already shown.
    fn render_query_input(&self, cx: &Context<Self>) -> Input {
        let query = self.query_input.read(cx).value();
        let showing_results = matches!(
            &self.search_state,
            SearchState::Ready(report)
                if report.query.as_ref() == query.trim() && report.mode == self.search_mode
        );
        let hint = if showing_results {
            Some("go to result")
        } else {
            enter_hint(&query)
        };
        let input = Input::new(&self.query_input).small();
        let Some(hint) = hint else {
            return input;
        };
        input.suffix(
            div()
                .debug_selector(|| "search-enter-hint".into())
                .flex_none()
                .text_xs()
                .text_color(cx.theme().muted_foreground)
                .child(format!("↵ {hint}")),
        )
    }

    pub(super) fn render_left_panel(&self, cx: &Context<Self>) -> impl IntoElement {
        let label_counts = self.node_label_counts.clone();
        let relationship_counts = self.relationship_counts.clone();
        let active_relationship = match self.active_facet.as_ref() {
            Some(FacetLens::Relationship(label)) => Some(label),
            Some(FacetLens::NodeLabel(_)) | None => None,
        };
        let active_node_label = match self.active_facet.as_ref() {
            Some(FacetLens::NodeLabel(label)) => Some(label),
            Some(FacetLens::Relationship(_)) | None => None,
        };
        let relationship_facets = visible_facet_counts(&relationship_counts, active_relationship);
        let node_label_facets = visible_facet_counts(&label_counts, active_node_label);
        let showing_search = !matches!(self.search_state, SearchState::Idle);
        let showing_path = !matches!(self.path_state, PathState::Idle);
        let showing_context = self.context_focus_active && !showing_search && !showing_path;
        let showing_facet = self.active_facet.is_some() && !showing_search && !showing_path;
        v_flex()
            .debug_selector(|| "left-panel".into())
            .w(px(self.left_panel_width()))
            .h_full()
            .flex_shrink_0()
            .border_r_1()
            .border_color(cx.theme().sidebar_border)
            .bg(cx.theme().sidebar)
            .child(self.render_bezel_sidebar_tabs(
                if showing_search {
                    Some("Search results")
                } else if showing_path {
                    Some("Evidence path")
                } else if showing_context {
                    Some("2-hop context")
                } else if showing_facet {
                    Some("Facet lens")
                } else {
                    None
                },
                cx,
            ))
            .when(showing_search, |this| {
                this.child(self.render_search_results(cx))
            })
            .when(showing_path, |this| {
                this.child(self.render_evidence_path(cx))
            })
            .when(!showing_search && !showing_path, |this| {
                this.child(section_label(
                    "RELATIONSHIPS",
                    relationship_counts.len(),
                    cx,
                ))
            })
            .when(!showing_search && !showing_path, |this| {
                this.child(
                    v_flex()
                        .id("relationship-facets")
                        .role(Role::List)
                        .aria_label("Relationship type facets")
                        .px(px(10.0))
                        .children(relationship_facets.iter().map(|(label, count)| {
                            let active = self.active_facet.as_ref()
                                == Some(&FacetLens::Relationship(label.clone()));
                            let facet = FacetLens::Relationship(label.clone());
                            Button::new(format!("relationship-facet:{label}"))
                                .ghost()
                                .small()
                                .compact()
                                .w_full()
                                .label(label.to_string())
                                .accessibility_id(format!("relationship-facet:{label}"))
                                .toggled(active)
                                .selected(active)
                                .cursor_pointer()
                                .when(active, |this| {
                                    this.border_l_2().border_color(cx.theme().ring)
                                })
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    this.activate_facet(facet.clone(), window, cx);
                                }))
                                .child(
                                    h_flex()
                                        .flex_1()
                                        .min_w_0()
                                        .gap_1()
                                        .items_center()
                                        .child(
                                            div()
                                                .size(px(7.0))
                                                .rounded_full()
                                                .bg(relationship_color(label)),
                                        )
                                        .child(div().flex_1())
                                        .when(active, |this| {
                                            this.child(
                                                div()
                                                    .font_family(
                                                        cx.theme().mono_font_family.clone(),
                                                    )
                                                    .text_xs()
                                                    .font_weight(gpui::FontWeight::BOLD)
                                                    .text_color(cx.theme().ring)
                                                    .child("LENS"),
                                            )
                                        })
                                        .when(!active, |this| {
                                            this.child(
                                                div()
                                                    .font_family(
                                                        cx.theme().mono_font_family.clone(),
                                                    )
                                                    .text_xs()
                                                    .text_color(cx.theme().muted_foreground)
                                                    .child(format_count(*count)),
                                            )
                                        }),
                                )
                        })),
                )
            })
            .when(!showing_search && !showing_path, |this| {
                this.child(section_label("NODE LABELS", label_counts.len(), cx))
            })
            .when(!showing_search && !showing_path, |this| {
                this.child(
                    v_flex()
                        .id("node-label-facets")
                        .role(Role::List)
                        .aria_label("Node label facets")
                        .px(px(10.0))
                        .children(node_label_facets.iter().map(|(label, count)| {
                            let active = self.active_facet.as_ref()
                                == Some(&FacetLens::NodeLabel(label.clone()));
                            let facet = FacetLens::NodeLabel(label.clone());
                            Button::new(format!("node-label-facet:{label}"))
                                .ghost()
                                .small()
                                .compact()
                                .w_full()
                                .label(label.to_string())
                                .accessibility_id(format!("node-label-facet:{label}"))
                                .toggled(active)
                                .selected(active)
                                .cursor_pointer()
                                .when(active, |this| {
                                    this.border_l_2().border_color(cx.theme().ring)
                                })
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    this.activate_facet(facet.clone(), window, cx);
                                }))
                                .child(
                                    h_flex()
                                        .flex_1()
                                        .min_w_0()
                                        .items_center()
                                        .child(div().flex_1())
                                        .when(active, |this| {
                                            this.child(
                                                div()
                                                    .mr_2()
                                                    .font_family(
                                                        cx.theme().mono_font_family.clone(),
                                                    )
                                                    .text_xs()
                                                    .font_weight(gpui::FontWeight::BOLD)
                                                    .text_color(cx.theme().ring)
                                                    .child("LENS"),
                                            )
                                        })
                                        .when(!active, |this| {
                                            this.child(
                                                div()
                                                    .font_family(
                                                        cx.theme().mono_font_family.clone(),
                                                    )
                                                    .text_xs()
                                                    .text_color(cx.theme().muted_foreground)
                                                    .child(format_count(*count)),
                                            )
                                        }),
                                )
                        }))
                        .children(self.hidden_label_rows(cx)),
                )
            })
    }

    fn left_panel_width(&self) -> f32 {
        if matches!(self.search_state, SearchState::Idle)
            && matches!(self.path_state, PathState::Idle)
        {
            OVERVIEW_SIDEBAR_WIDTH
        } else {
            DETAIL_SIDEBAR_WIDTH
        }
    }

    /// The ends of a path being traced, or that could not be, as the cards a
    /// found path shows them; by ID when the scene does not hold them.
    fn pending_path_ends(&self, start: u64, end: u64, cx: &Context<Self>) -> gpui::AnyElement {
        let card = |role, id, color| {
            let snapshot = self.snapshot()?;
            let index = snapshot.node_index(id)?;
            Some(
                path_endpoint_card_data(
                    role,
                    id,
                    &snapshot.nodes.labels[index],
                    &scene_node_title(snapshot, index),
                    color,
                    cx,
                )
                .into_any_element(),
            )
        };
        match (
            card("FROM", start, palette().celadon),
            card("TO", end, palette().copper),
        ) {
            (Some(from), Some(to)) => v_flex().gap_2().child(from).child(to).into_any_element(),
            _ => div()
                .font_family(cx.theme().mono_font_family.clone())
                .text_xs()
                .text_color(cx.theme().muted_foreground)
                .child(format!("node:{start}  →  node:{end}"))
                .into_any_element(),
        }
    }

    pub(super) fn render_evidence_path(&self, cx: &Context<Self>) -> gpui::AnyElement {
        match &self.path_state {
            PathState::Idle => div().into_any_element(),
            PathState::ChoosingEnd(draft) => {
                let origin = self.snapshot().and_then(|snapshot| {
                    snapshot.node_index(draft.start).map(|index| {
                        (
                            snapshot.nodes.labels[index].clone(),
                            scene_node_title(snapshot, index),
                        )
                    })
                });
                let destination = self.path_destination_candidate().and_then(|index| {
                    let snapshot = self.snapshot()?;
                    Some((
                        index,
                        *snapshot.nodes.ids.get(index)?,
                        snapshot.nodes.labels.get(index)?.clone(),
                        scene_node_title(snapshot, index),
                    ))
                });
                let has_destination = destination.is_some();
                v_flex()
                    .flex_1()
                    .min_h_0()
                    .p_4()
                    .gap_2()
                    .child(
                        h_flex()
                            .items_center()
                            .child(
                                div()
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(palette().celadon)
                                    .child("ORIGIN SET"),
                            )
                            .child(div().flex_1())
                            .child(
                                div()
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(if has_destination {
                                        "ENTER RUNS · ESC CANCELS"
                                    } else {
                                        "ESC CANCELS"
                                    }),
                            ),
                    )
                    .when_some(origin, |this, (label, title)| {
                        this.child(path_endpoint_card_data(
                            "FROM",
                            draft.start,
                            &label,
                            &title,
                            palette().celadon,
                            cx,
                        ))
                    })
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                h_flex()
                                    .items_center()
                                    .child(inspector_label("TRAVERSE"))
                                    .child(div().flex_1())
                                    .child(
                                        div()
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .child(path_direction_compact_label(draft.direction)),
                                    ),
                            )
                            .child(
                                h_flex()
                                    .gap_1()
                                    .child(
                                        Button::new("path-direction-both")
                                            .label("↔ Either")
                                            .small()
                                            .selected(draft.direction == PathDirection::Both)
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.set_path_direction(PathDirection::Both, cx);
                                            })),
                                    )
                                    .child(
                                        Button::new("path-direction-outgoing")
                                            .label("→ Out")
                                            .small()
                                            .selected(draft.direction == PathDirection::Outgoing)
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.set_path_direction(
                                                    PathDirection::Outgoing,
                                                    cx,
                                                );
                                            })),
                                    )
                                    .child(
                                        Button::new("path-direction-incoming")
                                            .label("← In")
                                            .small()
                                            .selected(draft.direction == PathDirection::Incoming)
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.set_path_direction(
                                                    PathDirection::Incoming,
                                                    cx,
                                                );
                                            })),
                                    ),
                            ),
                    )
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                h_flex()
                                    .items_center()
                                    .child(inspector_label("HOP LIMIT"))
                                    .child(div().flex_1())
                                    .child(
                                        div()
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .text_xs()
                                            .font_weight(gpui::FontWeight::BOLD)
                                            .text_color(palette().celadon)
                                            .child(format!("≤ {}", draft.max_hops)),
                                    ),
                            )
                            .child(
                                h_flex()
                                    .gap_1()
                                    .child(path_hop_button(1, draft.max_hops, cx))
                                    .child(path_hop_button(2, draft.max_hops, cx))
                                    .child(path_hop_button(4, draft.max_hops, cx))
                                    .child(path_hop_button(6, draft.max_hops, cx)),
                            ),
                    )
                    .when_some(destination, |this, (index, id, label, title)| {
                        this.child(
                            v_flex()
                                .pt_1()
                                .gap_2()
                                .child(path_endpoint_card_data(
                                    "TO",
                                    id,
                                    &label,
                                    &title,
                                    palette().copper,
                                    cx,
                                ))
                                .child(
                                    div()
                                        .debug_selector(|| "evidence-path-run".into())
                                        .w_full()
                                        .child(
                                            Button::new("evidence-path-run")
                                                .accessibility_id("evidence-path-run")
                                                .label("Trace exact path")
                                                .primary()
                                                .w_full()
                                                .on_click(cx.listener(
                                                    move |this, _, window, cx| {
                                                        this.trace_path_to_node(index, window, cx);
                                                    },
                                                )),
                                        ),
                                ),
                        )
                    })
                    .when(!has_destination, |this| {
                        this.child(
                            v_flex()
                                .gap_1()
                                .child(
                                    div()
                                        .text_sm()
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .child("Choose a destination"),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .line_height(px(17.0))
                                        .text_color(cx.theme().muted_foreground)
                                        .child(
                                            "Select another node on the canvas. Its exact-path action appears here.",
                                        ),
                                ),
                        )
                    })
                    .into_any_element()
            }
            PathState::Searching { start, end } => v_flex()
                .flex_1()
                .min_h_0()
                .p_4()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child("Tracing exact evidence path"),
                )
                .child(self.pending_path_ends(*start, *end, cx))
                .child(
                    div()
                        .text_xs()
                        .line_height(px(17.0))
                        .text_color(cx.theme().muted_foreground)
                        .child("Searching the complete database on a background worker…"),
                )
                .into_any_element(),
            PathState::Failed { start, end, error } => v_flex()
                .flex_1()
                .min_h_0()
                .p_4()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child("Could not trace evidence path"),
                )
                .child(self.pending_path_ends(*start, *end, cx))
                .child(
                    div()
                        .text_xs()
                        .line_height(px(17.0))
                        .text_color(cx.theme().warning)
                        .child(error.clone()),
                )
                .into_any_element(),
            PathState::Ready(report) => {
                let elapsed_ms = report.elapsed.as_secs_f64() * 1_000.0;
                let (outcome, outcome_color) = match report.termination {
                    EvidencePathTermination::Found => ("EXACT", palette().celadon),
                    EvidencePathTermination::NotFoundWithinHops => {
                        ("NO PATH", cx.theme().muted_foreground)
                    }
                    EvidencePathTermination::ExpansionLimit => ("INCOMPLETE", cx.theme().warning),
                };
                let hop_count = report.path.as_ref().map_or(0, |path| path.steps.len());
                let (visible_nodes, visible_edges) = self.snapshot().map_or((0, 0), |snapshot| {
                    report.path.as_ref().map_or_else(
                        || {
                            (
                                usize::from(snapshot.node_index(report.start.id).is_some())
                                    + usize::from(
                                        report.end.id != report.start.id
                                            && snapshot.node_index(report.end.id).is_some(),
                                    ),
                                0,
                            )
                        },
                        |path| {
                            (
                                path.nodes
                                    .iter()
                                    .filter(|node| snapshot.node_index(node.id).is_some())
                                    .count(),
                                path.steps
                                    .iter()
                                    .filter(|step| snapshot.edge_index(step.edge_id).is_some())
                                    .count(),
                            )
                        },
                    )
                });
                let total_nodes = report.path.as_ref().map_or(2, |path| path.nodes.len());
                let total_edges = report.path.as_ref().map_or(0, |path| path.steps.len());
                let partial = visible_nodes < total_nodes || visible_edges < total_edges;
                let retries = path_retries(report);
                let summary = match report.termination {
                    EvidencePathTermination::Found if hop_count == 0 => {
                        format!("Same node · {elapsed_ms:.1} ms")
                    }
                    EvidencePathTermination::Found => format!(
                        "{hop_count} hop{} · {elapsed_ms:.1} ms",
                        if hop_count == 1 { "" } else { "s" }
                    ),
                    EvidencePathTermination::NotFoundWithinHops => {
                        format!("None within {} hops · {elapsed_ms:.1} ms", report.max_hops)
                    }
                    EvidencePathTermination::ExpansionLimit => {
                        format!("Work cap reached · {elapsed_ms:.1} ms")
                    }
                };
                v_flex()
                    .flex_1()
                    .min_h_0()
                    .child(
                        v_flex()
                            .px_4()
                            .pt_3()
                            .pb_3()
                            .gap_2()
                            .border_b_1()
                            .border_color(cx.theme().border)
                            .child(
                                h_flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .text_xs()
                                            .font_weight(gpui::FontWeight::BOLD)
                                            .text_color(outcome_color)
                                            .child(outcome),
                                    )
                                    .child(div().flex_1())
                                    .child(
                                        div()
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .child(summary),
                                    )
                                    .when_some(evidence_path_text(report), |this, text| {
                                        this.child(
                                            div()
                                                .debug_selector(|| "evidence-path-copy".into())
                                                .child(
                                                    Button::new("evidence-path-copy")
                                                        .ghost()
                                                        .small()
                                                        .compact()
                                                        .label("Copy")
                                                        .tooltip("Copy the path as text")
                                                        .on_click(cx.listener(
                                                            move |this, _, _, cx| {
                                                                cx.write_to_clipboard(
                                                                    gpui::ClipboardItem::new_string(
                                                                        text.clone(),
                                                                    ),
                                                                );
                                                                this.status =
                                                                    "Copied the path as text".into();
                                                                cx.notify();
                                                            },
                                                        )),
                                                ),
                                        )
                                    }),
                            )
                            .child(path_endpoint_card("FROM", &report.start, palette().celadon, cx))
                            .child(path_endpoint_card("TO", &report.end, palette().copper, cx))
                            .child(path_plan_diagnostics(report, cx))
                            .when_some(report.relationship_label.clone(), |this, label| {
                                this.child(
                                    div()
                                        .font_family(cx.theme().mono_font_family.clone())
                                        .text_xs()
                                        .text_color(relationship_color(&label))
                                        .child(format!("relationship:{label}")),
                                )
                            })
                            .when(partial, |this| {
                                this.child(
                                    div()
                                        .px_2()
                                        .py_1()
                                        .rounded(px(4.0))
                                        .bg(cx.theme().warning.opacity(0.08))
                                        .text_xs()
                                        .line_height(px(16.0))
                                        .text_color(cx.theme().warning)
                                        .child(format!(
                                            "Exact database result; {} shows {visible_nodes}/{total_nodes} nodes and {visible_edges}/{total_edges} relationships.",
                                            if self.hidden_count() > 0 {
                                                "with nodes hidden, the canvas"
                                            } else {
                                                "sampled canvas"
                                            }
                                        )),
                                )
                            }),
                    )
                    .child(match &report.path {
                        Some(path) if path.steps.is_empty() => v_flex()
                            .flex_1()
                            .p_4()
                            .gap_1()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child("Zero-hop identity"),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .line_height(px(17.0))
                                    .text_color(cx.theme().muted_foreground)
                                    .child("The start and end resolve to the same graph node."),
                            )
                            .into_any_element(),
                        Some(path) => v_flex()
                            .id("evidence-path-steps")
                            .role(Role::List)
                            .aria_label("Ordered exact evidence path steps")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .px_2()
                            .py_1()
                            .children(path.steps.iter().enumerate().map(|(index, step)| {
                                let selected = self.snapshot().is_some_and(|snapshot| {
                                    snapshot.edge_index(step.edge_id).map(SceneSelection::Edge)
                                        == self.selection
                                });
                                let orientation = if step.forward {
                                    "stored direction"
                                } else {
                                    "reverse traversal"
                                };
                                let show_relationship_type =
                                    step.title.as_ref() != step.label.replace('_', " ");
                                let accessible_label = format!(
                                    "Evidence step {}, {}, node {} to node {}, relationship {}, {}",
                                    index + 1,
                                    step.title,
                                    step.from,
                                    step.to,
                                    step.label,
                                    orientation
                                );
                                let ring = cx.theme().ring;
                                let edge_id = step.edge_id;
                                div()
                                    .id(("evidence-path-step", step.edge_id))
                                    .debug_selector(move || format!("evidence-path-step-{index}"))
                                    .focusable()
                                    .tab_stop(true)
                                    .role(Role::Button)
                                    .aria_label(accessible_label)
                                    .aria_selected(selected)
                                    .w_full()
                                    .cursor_pointer()
                                    .mb_1()
                                    .px_2()
                                    .py_1()
                                    .rounded(px(5.0))
                                    .border_l_2()
                                    .border_color(relationship_color(&step.label))
                                    .when(selected, |this| {
                                        this.bg(cx.theme().sidebar_accent)
                                            .border_1()
                                            .border_color(cx.theme().ring)
                                    })
                                    .when(!selected, |this| {
                                        this.hover(|style| style.bg(cx.theme().sidebar_accent))
                                    })
                                    .focus(move |style| style.border_1().border_color(ring))
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.activate_evidence_step(index, window, cx);
                                    }))
                                    .on_hover(cx.listener(move |this, hovered: &bool, _, cx| {
                                        this.preview_edge(edge_id, *hovered, cx);
                                    }))
                                    .on_key_down(cx.listener(
                                        move |this, event: &gpui::KeyDownEvent, window, cx| {
                                            if !event.keystroke.modifiers.modified()
                                                && matches!(
                                                    event.keystroke.key.as_str(),
                                                    "enter" | "space"
                                                )
                                            {
                                                this.activate_evidence_step(index, window, cx);
                                                cx.stop_propagation();
                                            }
                                        },
                                    ))
                                    .child(
                                        h_flex()
                                            .w_full()
                                            .items_start()
                                            .gap_2()
                                            .child(
                                                div()
                                                    .w(px(22.0))
                                                    .flex_shrink_0()
                                                    .font_family(cx.theme().mono_font_family.clone())
                                                    .text_xs()
                                                    .font_weight(gpui::FontWeight::BOLD)
                                                    .text_color(relationship_color(&step.label))
                                                    .child(format!("{:02}", index + 1)),
                                            )
                                            .child(
                                                v_flex()
                                                    .flex_1()
                                                    .min_w_0()
                                                    .gap_0p5()
                                                    .child(
                                                        div()
                                                            .text_sm()
                                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                                            .truncate()
                                                            .child(step.title.to_string()),
                                                    )
                                                    .when(show_relationship_type, |this| {
                                                        this.child(
                                                            div()
                                                                .font_family(
                                                                    cx.theme()
                                                                        .mono_font_family
                                                                        .clone(),
                                                                )
                                                                .text_xs()
                                                                .text_color(relationship_color(
                                                                    &step.label,
                                                                ))
                                                                .truncate()
                                                                .child(step.label.to_string()),
                                                        )
                                                    })
                                                    .when_some(
                                                        evidence_step_names(path, index),
                                                        |this, names| {
                                                            this.child(
                                                                div()
                                                                    .text_xs()
                                                                    .truncate()
                                                                    .child(names),
                                                            )
                                                        },
                                                    )
                                                    .child(
                                                        div()
                                                            .font_family(cx.theme().mono_font_family.clone())
                                                            .text_xs()
                                                            .text_color(cx.theme().muted_foreground)
                                                            .child(format!(
                                                                "node:{} → node:{} · {}",
                                                                step.from, step.to, orientation
                                                            )),
                                                    ),
                                            ),
                                    )
                            }))
                            .into_any_element(),
                        None => v_flex()
                            .flex_1()
                            .p_4()
                            .gap_2()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child(match report.termination {
                                        EvidencePathTermination::NotFoundWithinHops => {
                                            "No evidence chain in this hop bound"
                                        }
                                        EvidencePathTermination::ExpansionLimit => {
                                            "Search stopped before it was conclusive"
                                        }
                                        EvidencePathTermination::Found => "No path payload",
                                    }),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .line_height(px(17.0))
                                    .text_color(match report.termination {
                                        EvidencePathTermination::ExpansionLimit => {
                                            cx.theme().warning
                                        }
                                        EvidencePathTermination::Found
                                        | EvidencePathTermination::NotFoundWithinHops => {
                                            cx.theme().muted_foreground
                                        }
                                    })
                                    .child(match report.termination {
                                        EvidencePathTermination::NotFoundWithinHops => format!(
                                            "The complete search found no matching chain up to {} hops.",
                                            report.max_hops
                                        ),
                                        EvidencePathTermination::ExpansionLimit => format!(
                                            "The {}-hop search reached its frontier-work budget. This is not proof of absence.",
                                            report.max_hops
                                        ),
                                        EvidencePathTermination::Found => {
                                            "The database returned an inconsistent path result.".into()
                                        }
                                    }),
                            )
                            .when(!retries.is_empty(), |this| {
                                this.child(
                                    h_flex()
                                        .flex_wrap()
                                        .gap_1()
                                        .children(retries.iter().map(|&retry| {
                                            path_retry_button(retry, cx)
                                        })),
                                )
                            })
                            .into_any_element(),
                    })
                    .into_any_element()
            }
        }
    }

    pub(super) fn render_search_results(&self, cx: &Context<Self>) -> gpui::AnyElement {
        match &self.search_state {
            SearchState::Idle => div().into_any_element(),
            SearchState::Searching { query, mode } => v_flex()
                .flex_1()
                .min_h_0()
                .p_4()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child(format!("{} search", mode.label())),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!(
                            "Searching all nodes and relationships for \"{query}\"…"
                        )),
                )
                .into_any_element(),
            SearchState::Failed { query, error } => v_flex()
                .flex_1()
                .min_h_0()
                .p_4()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child(format!("No results for \"{query}\"")),
                )
                .child(
                    div()
                        .text_xs()
                        .line_height(px(17.0))
                        .text_color(cx.theme().warning)
                        .child(error.clone()),
                )
                .into_any_element(),
            SearchState::Ready(report) => {
                let summary = format!(
                    "{} {} · {:.1} ms",
                    report.hits.len(),
                    report.mode.label().to_lowercase(),
                    report.elapsed.as_secs_f64() * 1_000.0
                );
                v_flex()
                    .flex_1()
                    .min_h_0()
                    .child(
                        v_flex()
                            .px_4()
                            .pt_3()
                            .pb_2()
                            .gap_1()
                            .child(
                                h_flex()
                                    .items_center()
                                    .child(
                                        div()
                                            .flex_1()
                                            .text_sm()
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .truncate()
                                            .child(format!("\"{}\"", report.query)),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .child(summary),
                                    ),
                            )
                            // Without the semantic half, naming the vectors
                            // would suggest they ranked these results.
                            .when_some(
                                report
                                    .embedding_model
                                    .clone()
                                    .filter(|_| report.warning.is_none()),
                                |this, model| {
                                    this.child(
                                        div()
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .truncate()
                                            .child(format!("Vectors · {model}")),
                                    )
                                },
                            )
                            // One line, with the reason on hover: it repeats
                            // on every search of a database the model cannot
                            // embed for.
                            .when_some(report.warning.clone(), |this, warning| {
                                let detail = warning.clone();
                                this.child(
                                    div()
                                        .id("search-warning")
                                        .text_xs()
                                        .truncate()
                                        .text_color(cx.theme().warning)
                                        .tooltip(move |window, cx| {
                                            Tooltip::text(detail.to_string(), window, cx)
                                        })
                                        .child(warning.to_string()),
                                )
                            }),
                    )
                    .child(
                        v_flex()
                            .id("search-results-scroll")
                            .role(Role::List)
                            .aria_label("Ranked graph search results")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .px_2()
                            .pb_3()
                            .when(report.hits.is_empty(), |this| {
                                this.child(
                                    div()
                                        .p_3()
                                        .text_sm()
                                        .text_color(cx.theme().muted_foreground)
                                        .child("No matching graph elements"),
                                )
                            })
                            .children(report.hits.iter().enumerate().map(|(index, hit)| {
                                let selected = index == self.selected_search_result;
                                let result_id = match hit.kind_label() {
                                    "EDGE" => ("search-result-edge", hit.id()),
                                    _ => ("search-result-node", hit.id()),
                                };
                                let kind_color = if hit.kind_label() == "EDGE" {
                                    relationship_color(&hit.label)
                                } else {
                                    palette().celadon
                                };
                                let accessible_label = format!(
                                    "{} {}, {}, relevance {:.0} percent",
                                    hit.kind_label().to_lowercase(),
                                    hit.title,
                                    hit.label,
                                    hit.score * 100.0
                                );
                                v_flex()
                                    .id(result_id)
                                    .role(Role::ListItem)
                                    .aria_label(accessible_label)
                                    .aria_selected(selected)
                                    .mx_0p5()
                                    .mb_1()
                                    .p_2()
                                    .gap_1()
                                    .rounded(px(5.0))
                                    .cursor_pointer()
                                    .when(selected, |this| {
                                        this.bg(cx.theme().sidebar_accent)
                                            .border_1()
                                            .border_color(cx.theme().ring)
                                    })
                                    .when(!selected, |this| {
                                        this.hover(|style| style.bg(cx.theme().sidebar_accent))
                                    })
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.activate_search_result(index, window, cx);
                                    }))
                                    .on_hover({
                                        let id = hit.id();
                                        let edge = hit.kind_label() == "EDGE";
                                        cx.listener(move |this, hovered: &bool, _, cx| {
                                            if edge {
                                                this.preview_edge(id, *hovered, cx);
                                            } else {
                                                this.preview_node(id, *hovered, cx);
                                            }
                                        })
                                    })
                                    .child(
                                        h_flex()
                                            .gap_2()
                                            .items_center()
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                                    .text_color(kind_color)
                                                    .child(hit.kind_label()),
                                            )
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .truncate()
                                                    .text_xs()
                                                    .text_color(cx.theme().muted_foreground)
                                                    .child(hit.label.to_string()),
                                            )
                                            .child(
                                                div()
                                                    .font_family(
                                                        cx.theme().mono_font_family.clone(),
                                                    )
                                                    .text_xs()
                                                    .text_color(cx.theme().muted_foreground)
                                                    .child(format!("{:.0}%", hit.score * 100.0)),
                                            ),
                                    )
                                    .child(
                                        title_line(&hit.title, cx)
                                            .text_sm()
                                            .font_weight(gpui::FontWeight::SEMIBOLD),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .line_height(px(16.0))
                                            .text_color(cx.theme().muted_foreground)
                                            .truncate()
                                            .child(hit.detail.to_string()),
                                    )
                                    .when(selected, |this| {
                                        this.child(
                                            v_flex()
                                                .mt_1()
                                                .gap_1()
                                                .when_some(hit.lexical_score, |this, score| {
                                                    this.child(search_signal(
                                                        "TEXT",
                                                        score,
                                                        palette().copper,
                                                        cx,
                                                    ))
                                                })
                                                .when_some(hit.semantic_score, |this, score| {
                                                    this.child(search_signal(
                                                        "VECTOR",
                                                        score,
                                                        palette().cobalt,
                                                        cx,
                                                    ))
                                                }),
                                        )
                                    })
                            })),
                    )
                    .into_any_element()
            }
        }
    }

    pub(super) fn render_inspector(&self, cx: &Context<Self>) -> impl IntoElement {
        let colors = palette();
        let header = h_flex()
            .items_center()
            .justify_between()
            .pr_2()
            .child(inspector_header_label("INSPECTOR"))
            .child(
                h_flex()
                    .gap_0p5()
                    .pt_2()
                    .child(self.copy_selection_button(cx))
                    .child(self.history_button(true, cx))
                    .child(self.history_button(false, cx)),
            );
        let content = match (self.snapshot(), self.selection) {
            (Some(snapshot), Some(SceneSelection::Node(index))) => {
                let properties = snapshot.nodes.properties[index].clone();
                let node_id = snapshot.nodes.ids[index];
                let path_origin = match self.path_state {
                    PathState::ChoosingEnd(draft) => Some(draft.start),
                    PathState::Idle
                    | PathState::Searching { .. }
                    | PathState::Ready(_)
                    | PathState::Failed { .. } => None,
                };
                let database_available = self.database_path.is_some();
                let pinned = self
                    .workspace
                    .as_ref()
                    .is_some_and(|workspace| workspace.borrow().is_pinned(index));
                v_flex()
                    .gap_3()
                    .px_3()
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                h_flex()
                                    .gap_2()
                                    .items_center()
                                    .text_xs()
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .child(copyable_id(
                                        "inspector-copy-node-id",
                                        format!("node:{node_id}"),
                                        colors.celadon,
                                        cx,
                                    ))
                                    .child(
                                        div()
                                            .text_color(cx.theme().muted_foreground)
                                            .truncate()
                                            .child(snapshot.nodes.labels[index].to_string()),
                                    ),
                            )
                            .child({
                                let title = scene_node_title(snapshot, index);
                                let (heading, directory) = match split_path_title(&title) {
                                    Some((name, parent)) => {
                                        (name.to_string(), Some(parent.to_string()))
                                    }
                                    None => (title.to_string(), None),
                                };
                                v_flex()
                                    .debug_selector(|| "inspector-node-title".into())
                                    .child(
                                        div()
                                            .text_lg()
                                            .line_height(px(24.0))
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .child(heading),
                                    )
                                    .when_some(directory, |this, directory| {
                                        this.child(
                                            div()
                                                .font_family(cx.theme().mono_font_family.clone())
                                                .text_xs()
                                                .text_color(cx.theme().muted_foreground)
                                                .child(directory),
                                        )
                                    })
                            })
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(format!(
                                        "{} relationships · {} vectors",
                                        snapshot.nodes.degrees[index],
                                        snapshot.nodes.vector_counts[index]
                                    )),
                            )
                            .when(pinned, |this| {
                                this.child(
                                    h_flex()
                                        .mt_2()
                                        .gap_2()
                                        .items_center()
                                        .child(div().size(px(6.0)).rounded_full().bg(colors.copper))
                                        .child(
                                            div()
                                                .text_xs()
                                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                                .child("Pinned to canvas"),
                                        )
                                        .child(div().flex_1())
                                        .child(
                                            Button::new("inspector-release-node")
                                                .label("Release")
                                                .small()
                                                .on_click(cx.listener(|this, _, _, cx| {
                                                    this.release_selected(cx);
                                                })),
                                        ),
                                )
                            }),
                    )
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(
                        v_flex()
                            .gap_2()
                            .child(inspector_label("PATH"))
                            .when_some(path_origin, |this, start| {
                                this.child(
                                    h_flex()
                                        .items_center()
                                        .gap_2()
                                        .child(
                                            div()
                                                .size(px(6.0))
                                                .rounded_full()
                                                .bg(palette().celadon),
                                        )
                                        .child(
                                            div()
                                                .font_family(cx.theme().mono_font_family.clone())
                                                .text_xs()
                                                .text_color(palette().celadon)
                                                .child(format!("ORIGIN · node:{start}")),
                                        ),
                                )
                            })
                            .child(
                                Button::new("inspector-path-start")
                                    .label(if path_origin == Some(node_id) {
                                        "Path origin"
                                    } else if path_origin.is_some() {
                                        "Move path origin here"
                                    } else {
                                        "Set as path origin"
                                    })
                                    .small()
                                    .disabled(!database_available)
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.choose_path_start(index, window, cx);
                                    })),
                            )
                            .when(path_origin == Some(node_id), |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .line_height(px(16.0))
                                        .text_color(cx.theme().muted_foreground)
                                        .child("Select another node on the canvas."),
                                )
                            })
                            .when_some(path_origin.filter(|&start| start != node_id), |this, _| {
                                this.child(
                                    Button::new("inspector-path-end")
                                        .label("Trace exact path")
                                        .small()
                                        .on_click(cx.listener(move |this, _, window, cx| {
                                            this.trace_path_to_node(index, window, cx);
                                        })),
                                )
                            })
                            .when(!database_available, |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .line_height(px(16.0))
                                        .text_color(cx.theme().muted_foreground)
                                        .child("Open a .vg database to trace exact paths."),
                                )
                            }),
                    )
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(self.render_node_connections(snapshot, index, cx))
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(
                        v_flex()
                            .gap_1()
                            .child(inspector_label("PROPERTIES"))
                            .when(properties.is_empty(), |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .text_color(cx.theme().muted_foreground)
                                        .child("No node properties"),
                                )
                            })
                            .children(
                                properties
                                    .iter()
                                    .take(INSPECTOR_PROPERTY_LIMIT)
                                    .enumerate()
                                    .map(|(position, property)| {
                                        property_row(position, &property.key, &property.value, cx)
                                    }),
                            )
                            .when(properties.len() > INSPECTOR_PROPERTY_LIMIT, |this| {
                                this.child(more_row(
                                    properties.len() - INSPECTOR_PROPERTY_LIMIT,
                                    "property",
                                    "properties",
                                    cx,
                                ))
                            }),
                    )
                    .into_any_element()
            }
            (Some(snapshot), Some(SceneSelection::Edge(index))) => {
                let source = snapshot.edges.sources[index] as usize;
                let target = snapshot.edges.targets[index] as usize;
                let label = &snapshot.edges.labels[index];
                let properties = snapshot.edges.properties[index].clone();
                v_flex()
                    .gap_3()
                    .px_3()
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                h_flex()
                                    .text_xs()
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .child(copyable_id(
                                        "inspector-copy-edge-id",
                                        format!("edge:{}", snapshot.edges.ids[index]),
                                        relationship_color(label),
                                        cx,
                                    )),
                            )
                            .child(
                                div()
                                    .text_lg()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child(label.to_string()),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(format!(
                                        "{} vector{} · directed relationship",
                                        snapshot.edges.vector_counts[index],
                                        if snapshot.edges.vector_counts[index] == 1 {
                                            ""
                                        } else {
                                            "s"
                                        }
                                    )),
                            ),
                    )
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(
                        v_flex()
                            .gap_2()
                            .child(inspector_label("DIRECTION"))
                            .child(endpoint_row("FROM", source, snapshot, cx))
                            .child(
                                div()
                                    .pl_2()
                                    .text_color(relationship_color(label))
                                    .text_sm()
                                    .child("↓"),
                            )
                            .child(endpoint_row("TO", target, snapshot, cx)),
                    )
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(
                        v_flex()
                            .gap_1()
                            .child(inspector_label("PROPERTIES"))
                            .when(properties.is_empty(), |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .text_color(cx.theme().muted_foreground)
                                        .child("No relationship properties"),
                                )
                            })
                            .children(
                                properties
                                    .iter()
                                    .take(INSPECTOR_PROPERTY_LIMIT)
                                    .enumerate()
                                    .map(|(position, property)| {
                                        property_row(position, &property.key, &property.value, cx)
                                    }),
                            )
                            .when(properties.len() > INSPECTOR_PROPERTY_LIMIT, |this| {
                                this.child(more_row(
                                    properties.len() - INSPECTOR_PROPERTY_LIMIT,
                                    "property",
                                    "properties",
                                    cx,
                                ))
                            }),
                    )
                    .into_any_element()
            }
            _ => {
                let theme = BezelTheme::of(cx).clone();
                v_flex()
                    .debug_selector(|| "inspector-empty-state".into())
                    .px_3()
                    .child(theme.empty_state(
                        icons::COMPASS,
                        "Nothing selected",
                        "Choose a node or relationship to inspect it.",
                    ))
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(
                        v_flex()
                            .pt_3()
                            .gap_2()
                            .child(inspector_shortcut(&theme, "↵", "Open connected context"))
                            .child(inspector_shortcut(&theme, "⌘K", "Search the graph"))
                            .child(inspector_shortcut(&theme, "esc", "Return to overview"))
                            .child(inspector_shortcut(&theme, "⌘O", "Open a database")),
                    )
                    .when_some(self.render_recent_databases(cx), |this, recent| {
                        this.child(recent)
                    })
                    .into_any_element()
            }
        };
        v_flex()
            .debug_selector(|| "inspector-panel".into())
            .w(px(296.0))
            .h_full()
            .flex_shrink_0()
            .border_l_1()
            .border_color(cx.theme().sidebar_border)
            .bg(cx.theme().sidebar)
            .child(header)
            .child(
                div()
                    .id("inspector-scroll")
                    .role(Role::Group)
                    .aria_label("Selected graph element properties")
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .pb_4()
                    .overflow_y_scroll()
                    .lock_scroll_axis()
                    .track_scroll(&self.inspector_scroll_handle)
                    .vertical_scrollbar(&self.inspector_scroll_handle)
                    .child(content),
            )
    }

    pub(super) fn render_canvas(
        &mut self,
        show_bezel_controls: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let entity = cx.entity().clone();
        let navigator_entity = entity.clone();
        let scene = match &self.state {
            LoadState::Ready(snapshot) => self.workspace.as_ref().map_or_else(
                || {
                    centered_state(
                        "Preparing workspace",
                        "Building interactive scene state.",
                        cx,
                    )
                },
                |workspace| {
                    div()
                        .absolute()
                        .size_full()
                        .child(graph_canvas(
                            snapshot.clone(),
                            workspace.clone(),
                            self.canvas_presentation(snapshot),
                        ))
                        .into_any_element()
                },
            ),
            LoadState::Loading { name } => centered_state(
                "Opening graph",
                format!("Mapping {name}, building a bounded scene, then laying it out."),
                cx,
            ),
            LoadState::Failed(error) => centered_state(
                "Could not open graph",
                format!("{error}\n⌘O opens another database."),
                cx,
            ),
        };
        let main_viewport = self
            .canvas_bounds
            .map_or(Vec2::new(800.0, 600.0), |bounds| {
                Vec2::new(bounds.size.width.into(), bounds.size.height.into())
            });
        let navigator = (self.semantic_depth().0 > 0)
            .then(|| {
                self.snapshot()
                    .cloned()
                    .zip(self.workspace.clone())
                    .map(|(snapshot, workspace)| {
                        graph_navigator(
                            snapshot,
                            workspace,
                            self.graph_navigator_cache.clone(),
                            GraphNavigatorPresentation {
                                camera: self.camera,
                                world_bounds: self.world_bounds,
                                main_viewport,
                                selection: self.selection,
                                emphasis: self.lens.as_ref().map(LensTransition::emphasis),
                            },
                        )
                        .into_any_element()
                    })
            })
            .flatten();
        div()
            .id("graph-canvas")
            .debug_selector(|| "graph-canvas".into())
            .role(Role::Group)
            .aria_label("Interactive graph canvas")
            .relative()
            .size_full()
            .overflow_hidden()
            .bg(palette().graphite)
            .track_focus(&self.focus_handle)
            .cursor_grab()
            .when(
                self.hovered_node.is_some() || self.hovered_edge.is_some(),
                |this| this.cursor_pointer(),
            )
            .when(self.drag.is_some(), |this| this.cursor_grabbing())
            .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                if !hovered {
                    this.set_hovered_node(None, cx);
                }
            }))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(Self::on_secondary_mouse_down),
            )
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_scroll_wheel(cx.listener(Self::on_scroll))
            .on_pinch(cx.listener(Self::on_pinch))
            .on_prepaint(move |bounds, _window, cx| {
                entity.update(cx, |this, _| {
                    this.canvas_bounds = Some(bounds);
                });
            })
            .child(scene)
            .children(self.render_fresh_rings())
            .children(self.render_hover_card(cx))
            .when_some(navigator, |this, navigator| {
                this.child(
                    div()
                        .id("graph-navigator")
                        .debug_selector(|| "graph-navigator".into())
                        .role(Role::Button)
                        .aria_label("Graph navigator; click to recenter the canvas")
                        .absolute()
                        .top(if show_bezel_controls {
                            px(12.0)
                        } else {
                            px(8.0)
                        })
                        .right(if show_bezel_controls {
                            px(12.0)
                        } else {
                            px(8.0)
                        })
                        .w(px(if show_bezel_controls { 148.0 } else { 116.0 }))
                        .h(px(if show_bezel_controls { 104.0 } else { 82.0 }))
                        .rounded(px(8.0))
                        .overflow_hidden()
                        .bg(rgb(0x10181e).opacity(0.93))
                        .border_1()
                        .border_color(cx.theme().border)
                        .shadow_sm()
                        .cursor_pointer()
                        .tooltip(|window, cx| {
                            Tooltip::text("Click to recenter the graph", window, cx)
                        })
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(Self::on_navigator_mouse_down),
                        )
                        .on_prepaint(move |bounds, _window, cx| {
                            navigator_entity.update(cx, |this, _| {
                                this.navigator_bounds = Some(bounds);
                            });
                        })
                        .child(navigator),
                )
            })
            .when(show_bezel_controls, |this| {
                this.child(
                    div()
                        .id("bezel-graph-controls")
                        .debug_selector(|| "bezel-graph-controls".into())
                        .absolute()
                        .bottom(px(18.0))
                        .left_0()
                        .right_0()
                        .px_3()
                        .flex()
                        .justify_center()
                        .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                        .child(
                            div()
                                .w_full()
                                .max_w(px(820.0))
                                .child(self.render_bezel_graph_controls(cx)),
                        ),
                )
            })
            .when(!show_bezel_controls, |this| {
                let (active_depth, level_label) = self.semantic_depth();
                this.child(
                    div()
                        .debug_selector(|| "compact-semantic-depth".into())
                        .absolute()
                        .left_3()
                        .bottom_3()
                        .px_2()
                        .py_1()
                        .rounded(px(4.0))
                        .bg(rgb(0x10181e).opacity(0.9))
                        .border_1()
                        .border_color(cx.theme().border)
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(h_flex().gap(px(3.0)).children((0..3).map(|index| {
                            div().w(px(8.0)).h(px(2.0)).rounded_full().bg(
                                if index <= active_depth {
                                    palette().celadon
                                } else {
                                    cx.theme().border
                                },
                            )
                        })))
                        .child(format!("{level_label}  {:.0}%", self.camera.zoom * 100.0)),
                )
            })
    }

    pub(super) fn render_status_bar(&self, cx: &Context<Self>) -> impl IntoElement {
        let (nodes, edges, sampled) = self
            .snapshot()
            .map(|snapshot| {
                (
                    snapshot.nodes.ids.len(),
                    snapshot.edges.ids.len(),
                    snapshot.sampled,
                )
            })
            .unwrap_or_default();
        let (pinned, moving) = self
            .workspace
            .as_ref()
            .map(|workspace| {
                let workspace = workspace.borrow();
                (workspace.pinned_count(), workspace.is_moving())
            })
            .unwrap_or_default();
        h_flex()
            .h(px(28.0))
            .flex_shrink_0()
            .px_3()
            .gap_3()
            .items_center()
            .border_t_1()
            .border_color(cx.theme().status_bar_border)
            .bg(cx.theme().status_bar)
            .font_family(cx.theme().mono_font_family.clone())
            .text_xs()
            .text_color(cx.theme().muted_foreground)
            .child(format!("{} nodes", format_count(nodes)))
            .child(format!("{} edges", format_count(edges)))
            .when(sampled, |this| this.child("sampled"))
            .child(format!(
                "{} layout",
                self.layout_kind.label().to_lowercase()
            ))
            .when(pinned > 0, |this| this.child(format!("{pinned} pinned")))
            .when(self.hidden_count() > 0, |this| {
                this.child(
                    div()
                        .id("show-hidden")
                        .debug_selector(|| "show-hidden".into())
                        .role(Role::Button)
                        .aria_label("Show hidden nodes")
                        .aria_keyshortcuts("Shift+Meta+H")
                        .px_1()
                        .rounded(px(4.0))
                        .cursor_pointer()
                        .text_color(palette().copper)
                        .hover(|this| this.bg(cx.theme().secondary_hover))
                        .tooltip(|window, cx| {
                            Tooltip::with_keystroke("Show hidden nodes", "⇧⌘H", window, cx)
                        })
                        .on_click(cx.listener(|this, _, _, cx| this.show_hidden_nodes(cx)))
                        .child(format!("{} hidden", self.hidden_count())),
                )
            })
            .when(moving, |this| this.child("settling"))
            .when(self.database_path.is_some(), |this| {
                this.child(self.render_live_refresh_toggle(cx))
            })
            .child(div().flex_1())
            .child(self.status.clone())
            .when_some(self.load_ms, |this, load_ms| {
                this.child(format!("{load_ms:.1} ms load+layout"))
            })
            .child(
                div()
                    .id("help-toggle")
                    .debug_selector(|| "help-toggle".into())
                    .role(Role::Button)
                    .aria_label("Shortcuts and commands")
                    .aria_keyshortcuts("Meta+Slash")
                    .px_1()
                    .rounded(px(4.0))
                    .cursor_pointer()
                    .hover(|this| this.bg(cx.theme().secondary_hover))
                    .tooltip(|window, cx| {
                        Tooltip::with_keystroke("Shortcuts and commands", "⌘/", window, cx)
                    })
                    .on_click(cx.listener(|this, _, window, cx| {
                        let visible = !this.help_visible;
                        this.set_help_visible(visible, window, cx);
                    }))
                    .child("?"),
            )
    }
}

impl StudioView {
    fn render_node_connections(
        &self,
        snapshot: &Arc<SceneSnapshot>,
        index: usize,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let connections = self.inspector_connections(snapshot, index);
        let mut shown_groups = 0;
        let mut hidden_groups = 0;
        let mut hidden_in_groups = 0;
        let mut groups = Vec::new();
        for group in &connections.groups {
            if shown_groups == INSPECTOR_CONNECTION_GROUPS {
                hidden_groups += 1;
                hidden_in_groups += group.total;
                continue;
            }
            shown_groups += 1;
            groups.push(group);
        }
        v_flex()
            .debug_selector(|| "inspector-connections".into())
            .gap_2()
            .child(
                h_flex()
                    .items_center()
                    .child(inspector_label("CONNECTIONS"))
                    .child(div().flex_1())
                    .child(
                        div()
                            .font_family(cx.theme().mono_font_family.clone())
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(format!("{} IN VIEW", format_count(connections.total))),
                    ),
            )
            .when(connections.total == 0, |this| {
                this.child(
                    div()
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child("No relationships in this view"),
                )
            })
            .children(groups.into_iter().map(|group| {
                let color = relationship_color(&group.label);
                let arrow = if group.outgoing { "→" } else { "←" };
                let key = (group.label.clone(), group.outgoing);
                let expanded = self.expanded_connections.contains(&key);
                let shown = group.connections.len().min(if expanded {
                    INSPECTOR_CONNECTIONS_EXPANDED
                } else {
                    INSPECTOR_CONNECTIONS_PER_GROUP
                });
                let selector = format!(
                    "{}:{}",
                    group.label,
                    if group.outgoing { "out" } else { "in" }
                );
                v_flex()
                    .gap_0p5()
                    .child(
                        h_flex()
                            .gap_1()
                            .items_center()
                            .font_family(cx.theme().mono_font_family.clone())
                            .text_xs()
                            .child(div().text_color(color).child(arrow))
                            .child(
                                div()
                                    .text_color(color)
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .truncate()
                                    .child(group.label.to_string()),
                            )
                            .child(
                                div()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(if group.outgoing { "out" } else { "in" }),
                            )
                            .child(div().flex_1())
                            .child(
                                div()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(format_count(group.total)),
                            ),
                    )
                    .children(group.connections.iter().take(shown).map(|connection| {
                        let neighbor = connection.neighbor;
                        let edge = connection.edge;
                        let edge_id = snapshot.edges.ids[edge];
                        let neighbor_id = snapshot.nodes.ids[neighbor];
                        let title = scene_node_title(snapshot, neighbor);
                        h_flex()
                            .id(("inspector-connection", edge_id))
                            .debug_selector(move || format!("inspector-connection:{neighbor_id}"))
                            .role(Role::Button)
                            .aria_label(format!(
                                "Select {} {} node {}",
                                group.label,
                                if group.outgoing { "target" } else { "source" },
                                title
                            ))
                            .gap_2()
                            .items_center()
                            .ml_3()
                            .px_1()
                            .py(px(1.0))
                            .rounded(px(4.0))
                            .cursor_pointer()
                            .hover(|style| style.bg(cx.theme().sidebar_accent))
                            .on_activate(cx, move |this, window, cx| {
                                this.select_related_node(neighbor, Some(edge), window, cx);
                            })
                            .on_hover(cx.listener(move |this, hovered: &bool, _, cx| {
                                this.preview_node(neighbor_id, *hovered, cx);
                            }))
                            .child(title_line(&title, cx).flex_1().text_sm())
                            .child(
                                div()
                                    .flex_shrink_0()
                                    .max_w(px(96.0))
                                    .truncate()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(snapshot.nodes.labels[neighbor].to_string()),
                            )
                    }))
                    .when(shown < group.connections.len(), |this| {
                        this.child(connections_toggle(
                            format!("inspector-connections-more:{selector}"),
                            format!("Show {} more", group.connections.len() - shown),
                            key.clone(),
                            cx,
                        ))
                    })
                    .when(group.total > group.connections.len() && expanded, |this| {
                        this.child(div().ml_3().px_1().child(more_row(
                            group.total - group.connections.len(),
                            "relationship",
                            "relationships",
                            cx,
                        )))
                    })
                    .when(expanded, |this| {
                        this.child(connections_toggle(
                            format!("inspector-connections-fewer:{selector}"),
                            "Show fewer".into(),
                            key,
                            cx,
                        ))
                    })
            }))
            .when(hidden_groups > 0, |this| {
                this.child(more_row(
                    hidden_in_groups,
                    "relationship",
                    "relationships",
                    cx,
                ))
            })
    }

    fn render_live_refresh_toggle(&self, cx: &Context<Self>) -> impl IntoElement {
        let colors = palette();
        let (marker, label, color) = if !self.live_refresh {
            ("○", "live off", cx.theme().muted_foreground)
        } else if self.reload_waiting {
            ("●", "updates waiting", colors.copper)
        } else if self.reload_in_progress {
            ("●", "refreshing", colors.celadon)
        } else {
            ("●", "live", colors.celadon)
        };
        h_flex()
            .id("live-refresh-toggle")
            .debug_selector(|| "live-refresh-toggle".into())
            .role(Role::Switch)
            .aria_label("Live refresh")
            .aria_toggled(if self.live_refresh {
                gpui::Toggled::True
            } else {
                gpui::Toggled::False
            })
            .gap_1()
            .px_1()
            .rounded(px(4.0))
            .cursor_pointer()
            .text_color(color)
            .hover(|this| this.bg(cx.theme().secondary_hover))
            .tooltip(|window, cx| Tooltip::with_keystroke("Toggle live refresh", "⇧⌘L", window, cx))
            .on_click(cx.listener(|this, _, window, cx| {
                let enabled = !this.live_refresh;
                this.set_live_refresh(enabled, window, cx);
            }))
            .child(marker)
            .child(label)
    }
}

impl StudioView {
    /// A small card beside the node under the pointer naming it, so the graph
    /// can be read without selecting every node. It follows the node while
    /// the layout moves and stays out of the way of a drag.
    /// The hovered node's card, and a ring that marks the node itself so a
    /// node pointed at from a list can be found on the canvas. Nothing is
    /// drawn for a node outside the visible canvas.
    fn render_hover_card(&self, cx: &Context<Self>) -> Vec<gpui::AnyElement> {
        if let Some(elements) = self.hover_card_elements(cx) {
            return Vec::from(elements);
        }
        self.edge_hover_elements(cx).unwrap_or_default()
    }

    /// The hovered relationship's card, opened where the pointer found it,
    /// and rings on whichever of its ends are on the canvas.
    fn edge_hover_elements(&self, cx: &Context<Self>) -> Option<Vec<gpui::AnyElement>> {
        const WIDTH: f32 = 280.0;
        const OFFSET: f32 = 14.0;
        if self.drag.is_some() {
            return None;
        }
        let hovered = self.hovered_edge?;
        let snapshot = self.snapshot()?;
        let edge = snapshot.edge_index(hovered.id)?;
        let bounds = self.canvas_bounds?;
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let source = snapshot.edges.sources[edge] as usize;
        let target = snapshot.edges.targets[edge] as usize;
        let mut elements: Vec<gpui::AnyElement> = {
            let workspace = self.workspace.as_ref()?.borrow();
            [source, target]
                .into_iter()
                .filter_map(|index| workspace.position(index))
                .map(|world| self.camera.project(world, self.world_bounds, viewport))
                .filter(|anchor| {
                    anchor.x >= 0.0
                        && anchor.y >= 0.0
                        && anchor.x <= viewport.x
                        && anchor.y <= viewport.y
                })
                .map(hover_ring)
                .collect()
        };
        let anchor = hovered.anchor;
        let left = if anchor.x + OFFSET + WIDTH <= viewport.x - 8.0 {
            anchor.x + OFFSET
        } else {
            (anchor.x - OFFSET - WIDTH).max(8.0)
        };
        let top = (anchor.y + OFFSET).clamp(8.0, (viewport.y - 84.0).max(8.0));
        let label = snapshot.edges.labels[edge].clone();
        let color = relationship_color(&label);
        elements.push(
            v_flex()
                .debug_selector(|| "graph-hover-edge-card".into())
                .absolute()
                .left(px(left))
                .top(px(top))
                .max_w(px(WIDTH))
                .px_2p5()
                .py_1p5()
                .gap_0p5()
                .rounded(px(8.0))
                .bg(rgb(0x10181e).opacity(0.94))
                .border_1()
                .border_color(cx.theme().border)
                .shadow_md()
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(color)
                        .child(label.to_string()),
                )
                .child(title_line(&scene_node_title(snapshot, source), cx).text_sm())
                .child(
                    h_flex()
                        .gap_1()
                        .min_w_0()
                        .child(div().text_sm().text_color(color).child("→"))
                        .child(
                            title_line(&scene_node_title(snapshot, target), cx)
                                .flex_1()
                                .text_sm(),
                        ),
                )
                .children(hover_properties(&snapshot.edges.properties[edge], "", cx))
                .into_any_element(),
        );
        Some(elements)
    }

    fn hover_card_elements(&self, cx: &Context<Self>) -> Option<[gpui::AnyElement; 2]> {
        const WIDTH: f32 = 260.0;
        const OFFSET: f32 = 14.0;
        if self.drag.is_some() {
            return None;
        }
        let snapshot = self.snapshot()?;
        let index = snapshot.node_index(self.hovered_node?)?;
        let bounds = self.canvas_bounds?;
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let world = self.workspace.as_ref()?.borrow().position(index)?;
        let anchor = self.camera.project(world, self.world_bounds, viewport);
        if anchor.x < 0.0 || anchor.y < 0.0 || anchor.x > viewport.x || anchor.y > viewport.y {
            return None;
        }
        let ring = hover_ring(anchor);
        // Open to the right of the node unless that would leave the canvas.
        let left = if anchor.x + OFFSET + WIDTH <= viewport.x - 8.0 {
            anchor.x + OFFSET
        } else {
            (anchor.x - OFFSET - WIDTH).max(8.0)
        };
        let top = (anchor.y - 18.0).clamp(8.0, (viewport.y - 64.0).max(8.0));
        let label = snapshot.nodes.labels[index].clone();
        let color = label_color(&label);
        let card = v_flex()
            .debug_selector(|| "graph-hover-card".into())
            .absolute()
            .left(px(left))
            .top(px(top))
            .max_w(px(WIDTH))
            .px_2p5()
            .py_1p5()
            .gap_0p5()
            .rounded(px(8.0))
            .bg(rgb(0x10181e).opacity(0.94))
            .border_1()
            .border_color(cx.theme().border)
            .shadow_md()
            .child(
                title_line(&scene_node_title(snapshot, index), cx)
                    .text_sm()
                    .font_weight(gpui::FontWeight::SEMIBOLD),
            )
            .child(
                h_flex()
                    .gap_1p5()
                    .items_center()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(div().size(px(6.0)).rounded_full().bg(color))
                    .child(label.to_string())
                    .child("·")
                    .child(format!(
                        "{} relationships",
                        format_count(snapshot.nodes.degrees[index] as usize)
                    )),
            )
            .children(hover_properties(
                &snapshot.nodes.properties[index],
                &scene_node_title(snapshot, index),
                cx,
            ))
            .into_any_element();
        Some([ring, card])
    }

    /// Rings on the nodes the last reload added, fading out, so new arrivals
    /// stand out while they spring into place.
    fn render_fresh_rings(&self) -> Vec<gpui::AnyElement> {
        const RING: f32 = 24.0;
        let (Some(fresh), Some(snapshot), Some(workspace), Some(bounds)) = (
            self.fresh_nodes.as_ref(),
            self.snapshot(),
            self.workspace.as_ref(),
            self.canvas_bounds,
        ) else {
            return Vec::new();
        };
        let opacity = fresh.opacity();
        let viewport = Vec2::new(bounds.size.width.into(), bounds.size.height.into());
        let workspace = workspace.borrow();
        fresh
            .ids
            .iter()
            .filter_map(|&id| workspace.position(snapshot.node_index(id)?))
            .map(|world| self.camera.project(world, self.world_bounds, viewport))
            .filter(|anchor| {
                anchor.x >= 0.0
                    && anchor.y >= 0.0
                    && anchor.x <= viewport.x
                    && anchor.y <= viewport.y
            })
            .map(|anchor| {
                div()
                    .debug_selector(|| "graph-fresh-ring".into())
                    .absolute()
                    .left(px(anchor.x - RING / 2.0))
                    .top(px(anchor.y - RING / 2.0))
                    .size(px(RING))
                    .rounded_full()
                    .border_2()
                    .border_color(palette().celadon.opacity(0.9 * opacity))
                    .bg(palette().celadon.opacity(0.12 * opacity))
                    .into_any_element()
            })
            .collect()
    }

    /// Recently opened databases, offered while nothing is selected.
    fn render_recent_databases(&self, cx: &Context<Self>) -> Option<gpui::AnyElement> {
        const SHOWN: usize = 5;
        let recent: Vec<PathBuf> = self.other_recent_databases().take(SHOWN).cloned().collect();
        if recent.is_empty() {
            return None;
        }
        Some(
            v_flex()
                .debug_selector(|| "inspector-recent".into())
                .pt_4()
                .gap_0p5()
                .child(div().pb_1().child(inspector_label("RECENT")))
                .children(recent.into_iter().enumerate().map(|(position, path)| {
                    let title = display_path(&path);
                    let label = format!("Open {title}");
                    h_flex()
                        .id(("inspector-recent", position))
                        .debug_selector(move || format!("inspector-recent:{position}"))
                        .role(Role::Button)
                        .aria_label(label)
                        .mx(px(-4.0))
                        .px_1()
                        .py(px(1.0))
                        .rounded(px(4.0))
                        .cursor_pointer()
                        .hover(|style| style.bg(cx.theme().sidebar_accent))
                        .on_activate(cx, move |this, window, cx| {
                            this.open_database(path.clone(), window, cx);
                        })
                        .child(title_line(&title, cx).flex_1().text_sm())
                }))
                .into_any_element(),
        )
    }

    fn copy_selection_button(&self, cx: &Context<Self>) -> impl IntoElement {
        let enabled = self.selection.is_some();
        div()
            .id("inspector-copy")
            .debug_selector(|| "inspector-copy".into())
            .role(Role::Button)
            .aria_label("Copy selection")
            .aria_keyshortcuts("Meta+C")
            .h(px(22.0))
            .px_1p5()
            .flex()
            .items_center()
            .rounded(px(4.0))
            .text_xs()
            .text_color(if enabled {
                cx.theme().foreground
            } else {
                cx.theme().muted_foreground.opacity(0.45)
            })
            .when(enabled, |this| {
                this.cursor_pointer()
                    .hover(|this| this.bg(cx.theme().secondary_hover))
                    .on_click(cx.listener(|this, _, _, cx| this.copy_selection(cx)))
            })
            .tooltip(|window, cx| Tooltip::with_keystroke("Copy as text", "⌘C", window, cx))
            .child("Copy")
    }

    fn history_button(&self, backward: bool, cx: &Context<Self>) -> impl IntoElement {
        let (id, glyph, label, keystroke, shortcut) = if backward {
            ("inspector-back", "←", "Back", "⌘[", "Meta+BracketLeft")
        } else {
            (
                "inspector-forward",
                "→",
                "Forward",
                "⌘]",
                "Meta+BracketRight",
            )
        };
        let enabled = if backward {
            !self.selection_back.is_empty()
        } else {
            !self.selection_forward.is_empty()
        };
        div()
            .id(id)
            .debug_selector(move || id.into())
            .role(Role::Button)
            .aria_label(label)
            .aria_keyshortcuts(shortcut)
            .size(px(22.0))
            .flex()
            .items_center()
            .justify_center()
            .rounded(px(4.0))
            .text_sm()
            .text_color(if enabled {
                cx.theme().foreground
            } else {
                cx.theme().muted_foreground.opacity(0.45)
            })
            .when(enabled, |this| {
                this.cursor_pointer()
                    .hover(|this| this.bg(cx.theme().secondary_hover))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.navigate_selection_history(backward, window, cx);
                    }))
            })
            .tooltip(move |window, cx| Tooltip::with_keystroke(label, keystroke, window, cx))
            .child(glyph)
    }
}

fn vecgra_logo_image() -> Arc<gpui::Image> {
    static LOGO: std::sync::OnceLock<Arc<gpui::Image>> = std::sync::OnceLock::new();
    LOGO.get_or_init(|| {
        Arc::new(gpui::Image::from_bytes(
            gpui::ImageFormat::Png,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/logo.png"
            ))
            .to_vec(),
        ))
    })
    .clone()
}

fn search_signal(
    label: &'static str,
    score: f32,
    color: gpui::Hsla,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    let score = score.clamp(0.0, 1.0);
    h_flex()
        .gap_2()
        .items_center()
        .font_family(cx.theme().mono_font_family.clone())
        .text_xs()
        .child(
            div()
                .w(px(44.0))
                .flex_shrink_0()
                .text_color(color)
                .child(label),
        )
        .child(
            div()
                .h(px(3.0))
                .flex_1()
                .overflow_hidden()
                .rounded_full()
                .bg(cx.theme().border)
                .child(div().h_full().w(relative(score)).rounded_full().bg(color)),
        )
        .child(
            div()
                .w(px(32.0))
                .flex_shrink_0()
                .text_right()
                .text_color(cx.theme().muted_foreground)
                .child(format!("{:.0}", score * 100.0)),
        )
}

fn path_endpoint_card(
    role: &'static str,
    node: &EvidenceNode,
    color: gpui::Hsla,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    path_endpoint_card_data(role, node.id, &node.label, &node.title, color, cx)
}

/// A path end: hovering points the node out on the canvas, and choosing it
/// selects it, when it is in the current scene.
fn path_endpoint_card_data(
    role: &'static str,
    id: u64,
    label: &str,
    title: &str,
    color: gpui::Hsla,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    let selector = format!("path-endpoint:{}", role.to_lowercase());
    h_flex()
        .id(SharedString::from(selector.clone()))
        .debug_selector(move || selector)
        .role(Role::Button)
        .aria_label(format!("Select {} node {title}", role.to_lowercase()))
        .mx(px(-4.0))
        .px(px(3.0))
        .py(px(1.0))
        .rounded(px(4.0))
        .cursor_pointer()
        .hover(|style| style.bg(cx.theme().sidebar_accent))
        .on_activate(cx, move |this, window, cx| {
            if let Some(index) = this.snapshot().and_then(|snapshot| snapshot.node_index(id)) {
                this.select_related_node(index, None, window, cx);
            }
        })
        .on_hover(cx.listener(move |this, hovered: &bool, _, cx| {
            this.preview_node(id, *hovered, cx);
        }))
        .gap_2()
        .items_center()
        .child(div().w(px(3.0)).h(px(30.0)).rounded_full().bg(color))
        .child(
            v_flex()
                .min_w_0()
                .gap_0p5()
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(color)
                                .child(role),
                        )
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .child(format!("node:{id} · {label}")),
                        ),
                )
                .child(
                    title_line(title, cx)
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD),
                ),
        )
}

/// A one-line element title. A path-like title leads with its last segment
/// and follows with the muted directory, so a list of files that truncates
/// still shows which file each row is.
fn title_line(title: &str, cx: &gpui::App) -> gpui::Div {
    match split_path_title(title) {
        Some((name, parent)) => h_flex()
            .min_w_0()
            .gap_1p5()
            .items_baseline()
            .child(
                div()
                    .flex_none()
                    .max_w(relative(0.8))
                    .truncate()
                    .child(name.to_string()),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .text_xs()
                    .font_weight(gpui::FontWeight::NORMAL)
                    .text_color(cx.theme().muted_foreground)
                    .child(parent.to_string()),
            ),
        None => div().min_w_0().truncate().child(title.to_string()),
    }
}

fn scene_node_title(snapshot: &SceneSnapshot, index: usize) -> Arc<str> {
    snapshot.node_title(index)
}

pub(super) const fn path_direction_label(direction: PathDirection) -> &'static str {
    match direction {
        PathDirection::Outgoing => "outgoing only",
        PathDirection::Incoming => "incoming only",
        PathDirection::Both => "either direction",
    }
}

const fn path_direction_compact_label(direction: PathDirection) -> &'static str {
    match direction {
        PathDirection::Outgoing => "FROM ORIGIN",
        PathDirection::Incoming => "TO ORIGIN",
        PathDirection::Both => "EITHER WAY",
    }
}

/// A found path as plain text to paste into notes or a prompt: one numbered
/// line per node, and between them the relationship, with `→` when it is
/// stored in path order and `←` when the path follows it backwards.
pub(super) fn evidence_path_text(report: &EvidencePathReport) -> Option<String> {
    let path = report.path.as_ref()?;
    let hops = path.steps.len();
    let mut text = format!(
        "Path from {} to {}: {}\n",
        report.start.title,
        report.end.title,
        match hops {
            0 => "same node".to_string(),
            1 => "1 hop".to_string(),
            hops => format!("{hops} hops"),
        }
    );
    for (index, node) in path.nodes.iter().enumerate() {
        text.push_str(&format!(
            "{}. {} ({}, node:{})\n",
            index + 1,
            node.title,
            node.label,
            node.id
        ));
        if let Some(step) = path.steps.get(index) {
            text.push_str(&format!(
                "   {} {} (relationship:{})\n",
                if step.forward { "→" } else { "←" },
                step.label,
                step.edge_id
            ));
        }
    }
    Some(text)
}

/// How many neighbors of each relationship group a copied node lists.
const COPIED_NEIGHBORS_PER_GROUP: usize = 20;

/// The selection as plain text for notes or a prompt: its title, label and
/// ID, its properties, and for a node its relationships in view, grouped as
/// the inspector groups them.
pub(super) fn selection_text(snapshot: &SceneSnapshot, selection: SceneSelection) -> String {
    let node = |index: usize| {
        format!(
            "{} ({}, node:{})",
            snapshot.node_title(index),
            snapshot.nodes.labels[index],
            snapshot.nodes.ids[index]
        )
    };
    let properties = |text: &mut String, properties: &[vecgra_studio_core::SceneProperty]| {
        for property in properties {
            if !matches!(property.value, PropertyValue::Bytes(_)) {
                text.push_str(&format!(
                    "{}: {}\n",
                    property.key,
                    format_value(&property.value)
                ));
            }
        }
    };
    match selection {
        SceneSelection::Node(index) => {
            let mut text = format!("{}\n", node(index));
            properties(&mut text, &snapshot.nodes.properties[index]);
            for group in snapshot
                .node_connections(index, COPIED_NEIGHBORS_PER_GROUP)
                .groups
            {
                let neighbors: Vec<String> = group
                    .connections
                    .iter()
                    .map(|connection| node(connection.neighbor))
                    .collect();
                let hidden = group.total - group.connections.len();
                text.push_str(&format!(
                    "{} {} ({}): {}{}\n",
                    if group.outgoing { "→" } else { "←" },
                    group.label,
                    group.total,
                    neighbors.join(", "),
                    if hidden > 0 {
                        format!(", and {hidden} more")
                    } else {
                        String::new()
                    }
                ));
            }
            text
        }
        SceneSelection::Edge(index) => {
            let mut text = format!(
                "{}: {} → {}\nrelationship:{}\n",
                snapshot.edges.labels[index],
                node(snapshot.edges.sources[index] as usize),
                node(snapshot.edges.targets[index] as usize),
                snapshot.edges.ids[index]
            );
            properties(&mut text, &snapshot.edges.properties[index]);
            text
        }
    }
}

/// A path step's two nodes by name, in path order, with file paths shortened
/// to their file name.
pub(super) fn evidence_step_names(
    path: &vecgra_studio_core::EvidencePath,
    index: usize,
) -> Option<String> {
    let short = |title: &str| {
        split_path_title(title)
            .map_or(title, |(name, _)| name)
            .to_owned()
    };
    let from = path.nodes.get(index)?;
    let to = path.nodes.get(index + 1)?;
    Some(format!("{} → {}", short(&from.title), short(&to.title)))
}

fn path_retry_button(retry: PathRetry, cx: &Context<StudioView>) -> impl IntoElement {
    let (selector, label) = match retry {
        PathRetry::EitherDirection => ("path-retry-direction", "Either direction".to_string()),
        PathRetry::AnyRelationship => ("path-retry-relationship", "Any relationship".to_string()),
        PathRetry::MoreHops(max_hops) => ("path-retry-hops", format!("Up to {max_hops} hops")),
    };
    div().debug_selector(move || selector.into()).child(
        Button::new(selector)
            .label(label)
            .small()
            .on_click(cx.listener(move |this, _, window, cx| {
                this.retry_evidence_path(retry, window, cx);
            })),
    )
}

fn path_hop_button(max_hops: usize, selected_max_hops: usize, cx: &Context<StudioView>) -> Button {
    Button::new(match max_hops {
        1 => "path-hops-1",
        2 => "path-hops-2",
        4 => "path-hops-4",
        6 => "path-hops-6",
        _ => "path-hops-custom",
    })
    .label(max_hops.to_string())
    .small()
    .selected(max_hops == selected_max_hops)
    .on_click(cx.listener(move |this, _, _, cx| {
        this.set_path_max_hops(max_hops, cx);
    }))
}

fn path_plan_diagnostics(
    report: &EvidencePathReport,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    v_flex()
        .gap_1()
        .child(
            h_flex()
                .items_center()
                .gap_2()
                .child(inspector_label("PLAN"))
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!(
                            "{} · {}",
                            path_strategy_label(report.strategy),
                            path_direction_compact_label(report.direction)
                        )),
                ),
        )
        .child(
            h_flex()
                .items_center()
                .gap_2()
                .child(
                    v_flex()
                        .gap_0p5()
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(palette().celadon)
                                .child("FROM"),
                        )
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .child(format!(
                                    "{} expanded",
                                    format_count(report.start_expanded_nodes)
                                )),
                        ),
                )
                .child(div().h(px(1.0)).flex_1().bg(cx.theme().border))
                .child(
                    v_flex()
                        .gap_0p5()
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_right()
                                .text_color(palette().copper)
                                .child("TO"),
                        )
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .text_xs()
                                .text_right()
                                .text_color(cx.theme().muted_foreground)
                                .child(format!(
                                    "{} expanded",
                                    format_count(report.end_expanded_nodes)
                                )),
                        ),
                ),
        )
        .child(
            h_flex()
                .gap_4()
                .child(path_work_stat("VISITED", report.visited_nodes, cx))
                .child(path_work_stat(
                    "REL READS",
                    report.examined_relationships,
                    cx,
                )),
        )
}

fn path_work_stat(label: &'static str, value: usize, cx: &Context<StudioView>) -> impl IntoElement {
    h_flex()
        .gap_1()
        .font_family(cx.theme().mono_font_family.clone())
        .text_xs()
        .child(div().text_color(cx.theme().muted_foreground).child(label))
        .child(
            div()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(format_count(value)),
        )
}

const fn path_strategy_label(strategy: EvidencePathStrategy) -> &'static str {
    match strategy {
        EvidencePathStrategy::BreadthFirst => "BREADTH-FIRST",
        EvidencePathStrategy::BidirectionalBreadthFirst => "BIDIRECTIONAL",
    }
}

fn section_label(label: &'static str, count: usize, cx: &Context<StudioView>) -> impl IntoElement {
    h_flex()
        .px_4()
        .pt_4()
        .pb_2()
        .items_center()
        .text_xs()
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .text_color(rgb(0x70838e))
        .child(label)
        .child(div().flex_1())
        .child(
            div()
                .font_family(cx.theme().mono_font_family.clone())
                .font_weight(gpui::FontWeight::NORMAL)
                .text_color(cx.theme().muted_foreground)
                .child(format!("{} TYPES", format_count(count))),
        )
}

fn inspector_shortcut(
    theme: &BezelTheme,
    shortcut: &'static str,
    description: &'static str,
) -> impl IntoElement {
    h_flex()
        .items_center()
        .gap_2()
        .child(popover::kbd_hint(theme, shortcut))
        .child(
            div()
                .text_xs()
                .text_color(theme.text_muted)
                .child(description),
        )
}

fn inspector_header_label(label: &'static str) -> impl IntoElement {
    div()
        .px_3()
        .pt_4()
        .pb_2()
        .text_xs()
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .text_color(rgb(0x70838e))
        .child(label)
}

fn inspector_label(label: &'static str) -> impl IntoElement {
    div()
        .text_xs()
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .text_color(rgb(0x70838e))
        .child(label)
}

/// The current time in Unix milliseconds, for reading dated properties.
fn unix_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| {
            i64::try_from(since.as_millis()).unwrap_or(i64::MAX)
        })
}

/// Up to three of an element's properties for a hover card, one line each.
fn hover_properties(
    properties: &[vecgra_studio_core::SceneProperty],
    title: &str,
    cx: &Context<StudioView>,
) -> Option<gpui::AnyElement> {
    let rows: Vec<_> = hover_property_lines(properties, title, unix_now_ms())
        .into_iter()
        .map(|(key, value)| {
            h_flex()
                .gap_1p5()
                .min_w_0()
                .text_xs()
                .child(
                    div()
                        .flex_none()
                        .text_color(cx.theme().muted_foreground)
                        .child(key.to_string()),
                )
                .child(
                    div()
                        .min_w_0()
                        .truncate()
                        .font_family(cx.theme().mono_font_family.clone())
                        .child(value),
                )
        })
        .collect();
    (!rows.is_empty()).then(|| {
        v_flex()
            .debug_selector(|| "graph-hover-properties".into())
            .pt_1()
            .gap_0p5()
            .children(rows)
            .into_any_element()
    })
}

/// The properties a hover card lists, as key and one-line value: up to
/// three, leaving out byte strings and the one already shown as the title.
/// Dates read as ages relative to `now_ms`, and a millisecond twin of a date
/// (`created_at_ms` beside `created_at`) is left out.
pub(super) fn hover_property_lines(
    properties: &[vecgra_studio_core::SceneProperty],
    title: &str,
    now_ms: i64,
) -> Vec<(Arc<str>, String)> {
    const SHOWN: usize = 3;
    let title_prefix = title.strip_suffix('…').unwrap_or(title);
    let is_date = |key: &str| {
        properties.iter().any(|property| {
            &*property.key == key
                && vecgra_studio_core::property_time_ms(key, &property.value).is_some()
        })
    };
    properties
        .iter()
        .filter_map(|property| {
            if matches!(property.value, PropertyValue::Bytes(_)) {
                return None;
            }
            if let Some(ms) = vecgra_studio_core::property_time_ms(&property.key, &property.value) {
                let twin = property.key.strip_suffix("_ms").is_some_and(is_date);
                return (!twin).then(|| {
                    (
                        property.key.clone(),
                        vecgra_studio_core::relative_age(ms, now_ms),
                    )
                });
            }
            let value = format_value(&property.value);
            let line = value.lines().map(str::trim).find(|line| !line.is_empty())?;
            let is_title = !title.is_empty() && (line == title || line.starts_with(title_prefix));
            (!is_title).then(|| (property.key.clone(), line.to_string()))
        })
        .take(SHOWN)
        .collect()
}

/// A ring around a node on the canvas, at its canvas-local position.
fn hover_ring(anchor: Vec2) -> gpui::AnyElement {
    const RING: f32 = 22.0;
    div()
        .debug_selector(|| "graph-hover-ring".into())
        .absolute()
        .left(px(anchor.x - RING / 2.0))
        .top(px(anchor.y - RING / 2.0))
        .size(px(RING))
        .rounded_full()
        .border_2()
        .border_color(palette().mist.opacity(0.85))
        .into_any_element()
}

/// Clickable rows that keyboard users can reach with Tab and activate with
/// Enter or Space, outlined while focused from the keyboard. The outline is a
/// border that is always there, so rows take a pixel less padding to keep
/// their height. A pointer click hands focus back to the graph, so Enter
/// afterwards still acts on the selection rather than on the clicked row.
trait ActivatableRow: Sized {
    fn on_activate(
        self,
        cx: &Context<StudioView>,
        activate: impl Fn(&mut StudioView, &mut Window, &mut Context<StudioView>) + 'static,
    ) -> Self;
}

impl ActivatableRow for gpui::Stateful<gpui::Div> {
    fn on_activate(
        self,
        cx: &Context<StudioView>,
        activate: impl Fn(&mut StudioView, &mut Window, &mut Context<StudioView>) + 'static,
    ) -> Self {
        let ring = cx.theme().ring;
        self.focusable()
            .tab_stop(true)
            .border_1()
            .border_color(gpui::transparent_black())
            .focus_visible(move |style| style.border_color(ring))
            .on_click(
                cx.listener(move |this, event: &gpui::ClickEvent, window, cx| {
                    if !matches!(event, gpui::ClickEvent::Keyboard(_)) {
                        window.focus(&this.focus_handle, cx);
                    }
                    activate(this, window, cx);
                }),
            )
    }
}

fn endpoint_row(
    direction: &'static str,
    index: usize,
    snapshot: &SceneSnapshot,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    let id = snapshot.nodes.ids[index];
    h_flex()
        .id(("inspector-endpoint", id))
        .role(Role::Button)
        .aria_label(format!(
            "Select {} endpoint node {id}",
            direction.to_lowercase()
        ))
        .gap_2()
        .items_center()
        .p(px(3.0))
        .rounded(px(4.0))
        .cursor_pointer()
        .hover(|style| style.bg(cx.theme().sidebar_accent))
        .on_activate(cx, move |this, window, cx| {
            this.select_related_node(index, None, window, cx);
        })
        .on_hover(cx.listener(move |this, hovered: &bool, _, cx| {
            this.preview_node(id, *hovered, cx);
        }))
        .child(
            div()
                .w(px(42.0))
                .flex_shrink_0()
                .text_xs()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(cx.theme().muted_foreground)
                .child(direction),
        )
        .child(
            v_flex()
                .min_w_0()
                .gap_0p5()
                .child(title_line(&scene_node_title(snapshot, index), cx).text_sm())
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .truncate()
                        .child(format!("node:{id} · {}", snapshot.nodes.labels[index])),
                ),
        )
}

fn copyable_id(
    element_id: &'static str,
    text: String,
    color: gpui::Hsla,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    let copied = text.clone();
    div()
        .id(element_id)
        .role(Role::Button)
        .aria_label(format!("Copy {text}"))
        .flex_shrink_0()
        .rounded(px(3.0))
        .cursor_pointer()
        .text_color(color)
        .hover(|style| style.underline())
        .tooltip(|window, cx| Tooltip::text("Copy ID", window, cx))
        .on_click(cx.listener(move |this, _, _, cx| {
            cx.write_to_clipboard(gpui::ClipboardItem::new_string(copied.clone()));
            this.status = format!("Copied {copied}").into();
            cx.notify();
        }))
        .child(text)
}

/// Expands or collapses one connection group in the inspector.
fn connections_toggle(
    selector: String,
    text: String,
    key: (Arc<str>, bool),
    cx: &Context<StudioView>,
) -> impl IntoElement {
    div()
        .id(SharedString::from(selector.clone()))
        .debug_selector(move || selector)
        .role(Role::Button)
        .ml_3()
        .px_1()
        .rounded(px(4.0))
        .text_xs()
        .text_color(cx.theme().muted_foreground)
        .cursor_pointer()
        .hover(|style| {
            style
                .bg(cx.theme().sidebar_accent)
                .text_color(cx.theme().foreground)
        })
        .on_activate(cx, move |this, _, cx| {
            if let Some(position) = this
                .expanded_connections
                .iter()
                .position(|expanded| *expanded == key)
            {
                this.expanded_connections.remove(position);
            } else {
                this.expanded_connections.push(key.clone());
            }
            cx.notify();
        })
        .child(text)
}

fn more_row(
    hidden: usize,
    singular: &'static str,
    plural: &'static str,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    div()
        .text_xs()
        .text_color(cx.theme().muted_foreground)
        .child(format!(
            "+{} more {}",
            format_count(hidden),
            if hidden == 1 { singular } else { plural }
        ))
}

/// One property in the inspector. Clicking it copies the value, at full
/// precision for floats; byte strings have nothing useful to copy.
fn property_row(
    position: usize,
    key: &str,
    value: &PropertyValue,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    let copy = match value {
        PropertyValue::Bytes(_) => None,
        PropertyValue::Float(value) => Some(value.to_string()),
        value => Some(format_value(value)),
    };
    let selector = format!("inspector-property:{key}");
    let age_selector = format!("inspector-property-age:{key}");
    let copied_key = key.to_string();
    let age = vecgra_studio_core::property_time_hint(key, value, unix_now_ms());
    v_flex()
        .id(("inspector-property", position))
        .debug_selector(move || selector)
        .gap_1()
        .px_2()
        .py(px(3.0))
        .mx(px(-8.0))
        .rounded(px(6.0))
        .when_some(copy, |this, copy| {
            this.role(Role::Button)
                .aria_label(format!("Copy {copied_key}"))
                .cursor_pointer()
                .hover(|style| style.bg(cx.theme().secondary_hover))
                .tooltip(|window, cx| Tooltip::text("Copy value", window, cx))
                .on_activate(cx, move |this, _, cx| {
                    cx.write_to_clipboard(gpui::ClipboardItem::new_string(copy.clone()));
                    this.status = format!("Copied {copied_key}").into();
                    cx.notify();
                })
        })
        .child(
            div()
                .text_xs()
                .text_color(cx.theme().muted_foreground)
                .child(key.to_string()),
        )
        .child(
            div()
                .font_family(cx.theme().mono_font_family.clone())
                .text_xs()
                .line_height(px(17.0))
                .child(format_value(value)),
        )
        .when_some(age, |this, age| {
            this.child(
                div()
                    .debug_selector(move || age_selector)
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(age),
            )
        })
}

fn format_value(value: &PropertyValue) -> String {
    match value {
        PropertyValue::Null => "null".into(),
        PropertyValue::Bool(value) => value.to_string(),
        PropertyValue::Int(value) => value.to_string(),
        PropertyValue::Float(value) => format!("{value:.5}"),
        PropertyValue::String(value) => value.to_string(),
        PropertyValue::Bytes(value) => format!("<{} bytes>", value.len()),
        PropertyValue::Node(value) => format!("node:{value}"),
        PropertyValue::Edge(value) => format!("edge:{value}"),
    }
}

fn centered_state(
    title: impl Into<SharedString>,
    detail: impl Into<SharedString>,
    cx: &Context<StudioView>,
) -> gpui::AnyElement {
    v_flex()
        .absolute()
        .size_full()
        .items_center()
        .justify_center()
        .gap_2()
        .child(
            div()
                .text_base()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(title.into()),
        )
        .child(
            div()
                .max_w(px(460.0))
                .text_sm()
                .text_color(cx.theme().muted_foreground)
                .child(detail.into()),
        )
        .into_any_element()
}

pub(super) fn format_count(value: usize) -> String {
    let text = value.to_string();
    let mut output = String::with_capacity(text.len() + text.len() / 3);
    for (index, character) in text.chars().enumerate() {
        if index > 0 && (text.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    output
}

pub(super) fn visible_facet_counts(
    counts: &[(Arc<str>, usize)],
    active: Option<&Arc<str>>,
) -> Vec<(Arc<str>, usize)> {
    const LIMIT: usize = 10;
    let mut visible: Vec<_> = counts.iter().take(LIMIT).cloned().collect();
    let Some(active) = active else {
        return visible;
    };
    if visible.iter().any(|(label, _)| label == active) {
        return visible;
    }
    let Some(active_count) = counts.iter().find(|(label, _)| label == active).cloned() else {
        return visible;
    };
    if visible.len() == LIMIT {
        visible.pop();
    }
    visible.insert(0, active_count);
    visible
}
