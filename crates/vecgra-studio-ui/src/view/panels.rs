use super::*;
use bezel_theme::Theme as BezelTheme;
use bezel_ui::{icons, popover, tooltip::Tooltip, widgets::Content as _};

#[cfg(target_os = "macos")]
const TITLE_BAR_PLATFORM_INSET: f32 = 80.0;
#[cfg(not(target_os = "macos"))]
const TITLE_BAR_PLATFORM_INSET: f32 = 12.0;
const TOOLBAR_GAP: f32 = 8.0;
const OVERVIEW_SIDEBAR_WIDTH: f32 = 218.0;
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
                                        .child(Input::new(&self.query_input).small()),
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
                            .child(Input::new(&self.query_input).small()),
                    )
                    .child(self.render_bezel_search_modes(cx))
                    .child(div().flex_1()),
            )
            .into_any_element()
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
                        })),
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
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!("node:{start}  →  node:{end}")),
                )
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
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!("node:{start}  →  node:{end}")),
                )
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
                                    ),
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
                                            "Exact database result; sampled canvas shows {visible_nodes}/{total_nodes} nodes and {visible_edges}/{total_edges} relationships."
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
                                div()
                                    .id(("evidence-path-step", step.edge_id))
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
                                            "The complete search found no matching chain up to {} hops. Increase max-hops to widen the proof boundary.",
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
                            "Searching all nodes and relationships for “{query}”…"
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
                        .child(format!("No results for “{query}”")),
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
                                            .child(format!("“{}”", report.query)),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .child(summary),
                                    ),
                            )
                            .when_some(report.embedding_model.clone(), |this, model| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .text_color(cx.theme().muted_foreground)
                                        .truncate()
                                        .child(format!("Vectors · {model}")),
                                )
                            })
                            .when_some(report.warning.clone(), |this, warning| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .line_height(px(16.0))
                                        .text_color(cx.theme().warning)
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
                                        div()
                                            .text_sm()
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .truncate()
                                            .child(hit.title.to_string()),
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
        let header = inspector_header_label("INSPECTOR");
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
                                div()
                                    .text_xs()
                                    .text_color(colors.celadon)
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .child(format!("node:{}", snapshot.nodes.ids[index])),
                            )
                            .child(
                                div()
                                    .text_lg()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child(snapshot.nodes.labels[index].to_string()),
                            )
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
                    .child(
                        v_flex()
                            .gap_2()
                            .child(inspector_label("PROPERTIES"))
                            .children(
                                properties.iter().take(24).map(|property| {
                                    property_row(&property.key, &property.value, cx)
                                }),
                            ),
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
                                div()
                                    .text_xs()
                                    .text_color(relationship_color(label))
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .child(format!("edge:{}", snapshot.edges.ids[index])),
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
                            .child(endpoint_row(
                                "FROM",
                                snapshot.nodes.ids[source],
                                &snapshot.nodes.labels[source],
                                cx,
                            ))
                            .child(
                                div()
                                    .pl_2()
                                    .text_color(relationship_color(label))
                                    .text_sm()
                                    .child("↓"),
                            )
                            .child(endpoint_row(
                                "TO",
                                snapshot.nodes.ids[target],
                                &snapshot.nodes.labels[target],
                                cx,
                            )),
                    )
                    .child(div().h(px(1.0)).bg(cx.theme().border))
                    .child(
                        v_flex()
                            .gap_2()
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
                                properties.iter().take(24).map(|property| {
                                    property_row(&property.key, &property.value, cx)
                                }),
                            ),
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
                            .child(inspector_shortcut(&theme, "esc", "Return to overview")),
                    )
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
        let dragging_node = match self.drag {
            Some(DragState::Node { index, .. }) => Some(index),
            Some(DragState::Canvas { .. }) | None => None,
        };
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
                            GraphCanvasPresentation {
                                camera: self.camera,
                                world_bounds: self.world_bounds,
                                selection: self.selection,
                                dragging: dragging_node,
                                emphasis: self.lens.as_ref().map(LensTransition::emphasis),
                                path_endpoints: self.path_endpoints,
                            },
                        ))
                        .into_any_element()
                },
            ),
            LoadState::Loading { name } => centered_state(
                "Opening graph",
                format!("Mapping {name}, building a bounded scene, then laying it out."),
                cx,
            ),
            LoadState::Failed(error) => centered_state("Could not open graph", error.clone(), cx),
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
            .when(self.drag.is_some(), |this| this.cursor_grabbing())
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
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
            .when(moving, |this| this.child("settling"))
            .child(div().flex_1())
            .child(self.status.clone())
            .when_some(self.load_ms, |this, load_ms| {
                this.child(format!("{load_ms:.1} ms load+layout"))
            })
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

fn path_endpoint_card_data(
    role: &'static str,
    id: u64,
    label: &str,
    title: &str,
    color: gpui::Hsla,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    h_flex()
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
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .truncate()
                        .child(title.to_string()),
                ),
        )
}

fn scene_node_title(snapshot: &SceneSnapshot, index: usize) -> Arc<str> {
    const PREFERRED_KEYS: [&str; 6] = ["title", "name", "headline", "path", "tag_name", "login"];
    snapshot.nodes.properties[index]
        .iter()
        .find_map(|property| {
            PREFERRED_KEYS
                .contains(&property.key.as_ref())
                .then_some(&property.value)
        })
        .and_then(|value| match value {
            PropertyValue::String(value) if !value.is_empty() => Some(value.clone()),
            _ => None,
        })
        .unwrap_or_else(|| snapshot.nodes.labels[index].clone())
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

fn endpoint_row(
    direction: &'static str,
    id: u64,
    label: &str,
    cx: &Context<StudioView>,
) -> impl IntoElement {
    h_flex()
        .gap_2()
        .items_center()
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
                .gap_0p5()
                .child(div().text_sm().child(label.to_string()))
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!("node:{id}")),
                ),
        )
}

fn property_row(key: &str, value: &PropertyValue, cx: &Context<StudioView>) -> impl IntoElement {
    v_flex()
        .gap_1()
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

fn format_count(value: usize) -> String {
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
