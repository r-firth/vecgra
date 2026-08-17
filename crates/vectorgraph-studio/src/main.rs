use std::path::PathBuf;

use gpui::{
    App, AppContext as _, KeyBinding, Styled as _, WindowBounds, WindowOptions, actions, px, size,
};
#[cfg(feature = "visual-test")]
use gpui::{Entity, Window};
use gpui_component::{ActiveTheme as _, Root, TitleBar};
use vectorgraph_studio_ui::{
    ArrangeAuto, ArrangeForce, ArrangeOrbit, ArrangeStructure, ClearSelection, FitView,
    FocusSearch, FocusSelectedContext, NextSearchResult, PreviousSearchResult, ReleaseSelected,
    StudioView, ZoomIn, ZoomOut, apply_studio_theme,
};

actions!(vectorgraph_studio_desktop, [Quit]);

fn main() {
    let database_path = std::env::args_os().nth(1).map(PathBuf::from);
    #[cfg(feature = "visual-test")]
    let capture_path = std::env::var_os("VG_STUDIO_CAPTURE").map(PathBuf::from);
    #[cfg(feature = "visual-test")]
    let capture_command = std::env::var("VG_STUDIO_CAPTURE_COMMAND").ok();
    #[cfg(feature = "visual-test")]
    let capture_result = std::env::var("VG_STUDIO_CAPTURE_RESULT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    #[cfg(feature = "visual-test")]
    let capture_zoom = std::env::var("VG_STUDIO_CAPTURE_ZOOM")
        .ok()
        .and_then(|value| value.parse::<f32>().ok());
    #[cfg(feature = "visual-test")]
    let capture_center = std::env::var("VG_STUDIO_CAPTURE_CENTER")
        .ok()
        .and_then(|value| value.parse::<u64>().ok());
    let app = gpui_platform::application().with_assets(gpui_component_assets::Assets);

    app.run(move |cx| {
        cx.set_app_identity("dev.vectorgraph.studio", "VectorGraph Studio");
        gpui_component::init(cx);
        apply_studio_theme(cx);

        cx.bind_keys([
            #[cfg(target_os = "macos")]
            KeyBinding::new("cmd-q", Quit, None),
            #[cfg(not(target_os = "macos"))]
            KeyBinding::new("alt-f4", Quit, None),
            KeyBinding::new("cmd-0", FitView, Some("VectorGraphStudio")),
            KeyBinding::new("cmd-=", ZoomIn, Some("VectorGraphStudio")),
            KeyBinding::new("cmd--", ZoomOut, Some("VectorGraphStudio")),
            KeyBinding::new("cmd-1", ArrangeAuto, Some("VectorGraphStudio")),
            KeyBinding::new("cmd-2", ArrangeForce, Some("VectorGraphStudio")),
            KeyBinding::new("cmd-3", ArrangeOrbit, Some("VectorGraphStudio")),
            KeyBinding::new("cmd-4", ArrangeStructure, Some("VectorGraphStudio")),
            KeyBinding::new("cmd-shift-r", ReleaseSelected, Some("VectorGraphStudio")),
            KeyBinding::new("enter", FocusSelectedContext, Some("VectorGraphStudio")),
            KeyBinding::new("cmd-k", FocusSearch, Some("VectorGraphStudio")),
            KeyBinding::new("down", NextSearchResult, Some("VectorGraphStudio")),
            KeyBinding::new("up", PreviousSearchResult, Some("VectorGraphStudio")),
            KeyBinding::new("escape", ClearSelection, Some("VectorGraphStudio")),
        ]);
        cx.on_action(|_: &Quit, cx: &mut App| cx.quit());

        let mut window_options = WindowOptions {
            window_bounds: Some(WindowBounds::centered(size(px(1_340.0), px(820.0)), cx)),
            window_min_size: Some(size(px(760.0), px(520.0))),
            app_id: Some("dev.vectorgraph.studio".into()),
            ..TitleBar::window_options()
        };
        window_options.focus = true;

        cx.spawn(async move |cx| {
            cx.open_window(window_options, move |window, cx| {
                window.activate_window();
                window.set_window_title("VectorGraph Studio");
                #[cfg(feature = "visual-test")]
                let waiting_for_database = database_path.is_some();
                let view = cx.new(|cx| StudioView::new(database_path, window, cx));
                let root =
                    cx.new(|cx| Root::new(view.clone(), window, cx).bg(cx.theme().background));

                #[cfg(feature = "visual-test")]
                if let Some(capture_path) = capture_path {
                    let capture_view = view.clone();
                    if waiting_for_database && !view.read(cx).is_ready() {
                        view.update(cx, |view, _| {
                            view.set_on_ready(move |window, cx| {
                                schedule_capture(
                                    window,
                                    cx,
                                    capture_path,
                                    capture_view,
                                    capture_command,
                                    capture_result,
                                    capture_zoom,
                                    capture_center,
                                );
                            });
                        });
                    } else {
                        schedule_capture(
                            window,
                            cx,
                            capture_path,
                            capture_view,
                            capture_command,
                            capture_result,
                            capture_zoom,
                            capture_center,
                        );
                    }
                }

                root
            })
            .expect("open the VectorGraph Studio window");
        })
        .detach();
    });
}

#[cfg(feature = "visual-test")]
fn schedule_capture(
    window: &Window,
    cx: &mut App,
    capture_path: PathBuf,
    view: Entity<StudioView>,
    capture_command: Option<String>,
    capture_result: Option<usize>,
    capture_zoom: Option<f32>,
    capture_center: Option<u64>,
) {
    window.defer(cx, move |window, cx| {
        if let Some(command) = capture_command {
            view.update(cx, |view, cx| {
                view.execute_command(&command, window, cx);
            });
            if view.read(cx).is_searching() {
                let capture_view = view.clone();
                view.update(cx, |view, _| {
                    view.set_on_search_ready(move |window, cx| {
                        window.defer(cx, move |window, cx| {
                            finish_visual_state(
                                window,
                                cx,
                                capture_path,
                                capture_view,
                                capture_result,
                                capture_zoom,
                                capture_center,
                            );
                        });
                    });
                });
                return;
            }
            if view.read(cx).is_arranging() {
                let capture_view = view.clone();
                view.update(cx, |view, _| {
                    view.set_on_layout_ready(move |window, cx| {
                        window.defer(cx, move |window, cx| {
                            finish_visual_state(
                                window,
                                cx,
                                capture_path,
                                capture_view,
                                capture_result,
                                capture_zoom,
                                capture_center,
                            );
                        });
                    });
                });
                return;
            }
        }
        finish_visual_state(
            window,
            cx,
            capture_path,
            view,
            capture_result,
            capture_zoom,
            capture_center,
        );
    });
}

#[cfg(feature = "visual-test")]
fn finish_visual_state(
    window: &mut Window,
    cx: &mut App,
    capture_path: PathBuf,
    view: Entity<StudioView>,
    capture_result: Option<usize>,
    capture_zoom: Option<f32>,
    capture_center: Option<u64>,
) {
    // Finish any load/layout transition before applying a capture-specific
    // camera center. Otherwise the camera follows the node's old presentation
    // position while its spring moves to the new arrangement.
    view.update(cx, |view, _| view.settle_presentation_for_capture());
    if let Some(index) = capture_result {
        view.update(cx, |view, cx| {
            view.activate_search_result(index, window, cx);
        });
    }
    if let Some(node_id) = capture_center {
        view.update(cx, |view, cx| {
            view.execute_command(&format!("center {node_id}"), window, cx);
        });
    }
    if let Some(zoom) = capture_zoom {
        view.update(cx, |view, cx| {
            view.execute_command(&format!("zoom {zoom}"), window, cx);
        });
    }
    view.update(cx, |view, _| view.settle_presentation_for_capture());
    capture_frame(window, cx, capture_path);
}

#[cfg(feature = "visual-test")]
fn capture_frame(window: &mut Window, cx: &mut App, capture_path: PathBuf) {
    let arena = window.draw(cx);
    match window.render_to_image() {
        Ok(image) => image
            .save(&capture_path)
            .expect("save the VectorGraph Studio visual-test capture"),
        Err(error) => panic!("render the VectorGraph Studio capture: {error}"),
    }
    let frames = window.frame_duration_snapshot();
    let draws = &frames.draw_duration_histogram;
    if !draws.is_empty() {
        eprintln!(
            "studio_draw_ms\tcount={}\tp50={:.3}\tp95={:.3}\tmax={:.3}",
            draws.len(),
            draws.value_at_quantile(0.50) as f64 / 1_000_000.0,
            draws.value_at_quantile(0.95) as f64 / 1_000_000.0,
            draws.max() as f64 / 1_000_000.0,
        );
    }
    arena.clear(cx);
    cx.quit();
}
