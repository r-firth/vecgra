//! The canvas as an SVG document, so a view can be shared outside Studio.

use std::fmt::Write as _;

use super::*;

/// The canvas as an SVG document `viewport` pixels across, drawn the way the
/// canvas draws it: the background, relationships, nodes, then names.
pub(crate) fn scene_svg(
    snapshot: &SceneSnapshot,
    workspace: &GraphWorkspace,
    presentation: &GraphCanvasPresentation,
    viewport: Vec2,
) -> String {
    let bounds = Bounds {
        origin: point(px(0.0), px(0.0)),
        size: size(px(viewport.x), px(viewport.y)),
    };
    let emphasis = presentation.emphasis.as_ref();
    let scene = prepare_scene(
        snapshot,
        workspace,
        ScenePresentation {
            camera: presentation.camera,
            world_bounds: presentation.world_bounds,
            selection: presentation.selection,
            dragging: None,
            emphasis,
            path_endpoints: presentation.path_endpoints,
            hovered_node: None,
            hovered_edge: None,
            label_colors: &presentation.label_colors,
        },
        bounds,
    );
    let projected: Vec<Vec2> = workspace
        .positions()
        .iter()
        .map(|&position| {
            presentation
                .camera
                .project(position, presentation.world_bounds, viewport)
        })
        .collect();
    let colors = palette();
    let (width, height) = (viewport.x.round(), viewport.y.round());
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Inter, system-ui, sans-serif">"#
    );
    let _ = writeln!(
        svg,
        r#"<rect width="{width}" height="{height}" fill="{}"/>"#,
        hex(colors.graphite)
    );

    // Relationships, weighted as the canvas weights them: the selected one,
    // then those a lens emphasizes, then the selection's own, then the rest.
    let selected_node = presentation.selection.and_then(SceneSelection::node);
    let selected_edge = presentation.selection.and_then(SceneSelection::edge);
    let dim = emphasis.map_or(0.0, |lens| lens.dim);
    svg.push_str("<g stroke-linecap=\"round\">\n");
    for edge in 0..snapshot.edges.ids.len() {
        let source = snapshot.edges.sources[edge] as usize;
        let target = snapshot.edges.targets[edge] as usize;
        let (from, to) = (projected[source], projected[target]);
        if !segment_may_be_visible(from, to, viewport) {
            continue;
        }
        let score = emphasis.map_or(0.0, |lens| lens.edge_score(edge));
        let incident = selected_node.is_some_and(|node| node == source || node == target);
        let (stroke, opacity) = if selected_edge == Some(edge) {
            (2.5, 1.0)
        } else if score > 0.01 {
            (1.2 + score * 1.5, 0.42 + score * 0.58)
        } else if incident && dim < 0.35 {
            (1.65, 0.78)
        } else {
            (0.95, 0.27 * (1.0 - dim).powf(1.6))
        };
        if opacity < 0.01 {
            continue;
        }
        let _ = writeln!(
            svg,
            r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-opacity="{opacity:.2}" stroke-width="{stroke:.2}"/>"#,
            from.x,
            from.y,
            to.x,
            to.y,
            hex(relationship_color(&snapshot.edges.labels[edge]))
        );
    }
    svg.push_str("</g>\n");

    for arrow in &scene.arrows {
        let [tip, left, right] = arrow
            .corners
            .map(|corner| (f32::from(corner.x), f32::from(corner.y)));
        let _ = writeln!(
            svg,
            r#"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="{}" fill-opacity="{:.2}"/>"#,
            tip.0,
            tip.1,
            left.0,
            left.1,
            right.0,
            right.1,
            hex(arrow.color),
            arrow.color.a * if arrow.selected { 1.0 } else { 0.88 }
        );
    }

    for caption in &scene.captions {
        let text_width = caption.text.chars().count() as f32 * 6.2;
        let (x, y) = (f32::from(caption.center.x), f32::from(caption.center.y));
        let _ = writeln!(
            svg,
            r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="18" rx="5" fill="{}" fill-opacity="0.9" stroke="{}" stroke-opacity="{}"/>"#,
            x - text_width / 2.0 - 6.0,
            y - 9.0,
            text_width + 12.0,
            hex(colors.graphite),
            hex(caption.color),
            if caption.selected { 1.0 } else { 0.75 }
        );
        let _ = writeln!(
            svg,
            r#"<text x="{x:.1}" y="{:.1}" font-size="10" text-anchor="middle" fill="{}">{}</text>"#,
            y + 3.5,
            hex(colors.mist),
            escape(&caption.text)
        );
    }

    for node in &scene.nodes {
        let (x, y) = (f32::from(node.point.x), f32::from(node.point.y));
        let fill = hex(node.color);
        let fill_opacity = node.color.a;
        let ring = if node.path_start && node.path_end {
            Some(colors.mist)
        } else if node.path_start {
            Some(colors.celadon)
        } else if node.path_end {
            Some(colors.copper)
        } else if node.selected {
            Some(colors.mist)
        } else if node.pinned {
            Some(colors.copper)
        } else {
            None
        };
        if node.selected || node.path_start || node.path_end {
            let _ = writeln!(
                svg,
                r#"<circle cx="{x:.1}" cy="{y:.1}" r="{:.1}" fill="{}" fill-opacity="0.12"/>"#,
                node.radius + 10.0,
                hex(ring.unwrap_or(colors.cobalt))
            );
        }
        match ring {
            Some(ring) => {
                let _ = writeln!(
                    svg,
                    r#"<circle cx="{x:.1}" cy="{y:.1}" r="{:.1}" fill="{}" fill-opacity="{fill_opacity:.2}" stroke="{}" stroke-width="1.5"/>"#,
                    node.radius + 2.0,
                    if node.selected {
                        hex(colors.celadon)
                    } else {
                        fill
                    },
                    hex(ring)
                );
            }
            None => {
                let _ = writeln!(
                    svg,
                    r#"<circle cx="{x:.1}" cy="{y:.1}" r="{:.1}" fill="{fill}" fill-opacity="{fill_opacity:.2}"/>"#,
                    node.radius
                );
            }
        }
    }

    for caption in &scene.node_captions {
        let (x, y) = (f32::from(caption.origin.x), f32::from(caption.origin.y));
        let _ = writeln!(
            svg,
            r#"<text x="{:.1}" y="{:.1}" font-size="10.5" font-weight="{}" fill="{}" fill-opacity="{}" stroke="{}" stroke-opacity="0.72" stroke-width="3" paint-order="stroke">{}</text>"#,
            x + 4.0,
            y + 12.0,
            if caption.strong { 600 } else { 400 },
            hex(colors.mist),
            if caption.strong { 1.0 } else { 0.86 },
            hex(colors.graphite),
            escape(&caption.text)
        );
    }
    svg.push_str("</svg>\n");
    svg
}

/// A color as `#rrggbb`; its alpha is written separately as an opacity.
fn hex(color: Hsla) -> String {
    let rgb = color.to_rgb();
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!(
        "#{:02x}{:02x}{:02x}",
        channel(rgb.r),
        channel(rgb.g),
        channel(rgb.b)
    )
}

fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            character => escaped.push(character),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_view_exports_as_svg_with_its_names() {
        let mut snapshot = SceneSnapshot::demo();
        snapshot.layout(vecgra_studio_core::LayoutOptions::default());
        snapshot.nodes.properties[5] = vec![vecgra_studio_core::SceneProperty {
            key: "title".into(),
            value: vecgra_studio_core::PropertyValue::String("R&D <notes>".into()),
        }]
        .into();
        let workspace = GraphWorkspace::new(&snapshot);
        let viewport = Vec2::new(1_200.0, 900.0);
        let svg = scene_svg(
            &snapshot,
            &workspace,
            &GraphCanvasPresentation {
                camera: Camera::fit(snapshot.bounds),
                world_bounds: snapshot.bounds,
                selection: Some(SceneSelection::Node(5)),
                dragging: None,
                emphasis: None,
                path_endpoints: None,
                hovered_node: None,
                hovered_edge: None,
                label_colors: Default::default(),
            },
            viewport,
        );
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\""));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert_eq!(svg.matches("<line ").count(), snapshot.edges.ids.len());
        // Every node is drawn, and the selection also gets a halo.
        assert_eq!(
            svg.matches("<circle ").count(),
            snapshot.nodes.ids.len() + 1
        );
        // The selection's relationships keep the arrowheads that show their direction.
        assert!(svg.contains("<polygon points="));
        assert!(svg.contains(">R&amp;D &lt;notes&gt;</text>"));
        assert!(!svg.contains("R&D"));
    }
}
