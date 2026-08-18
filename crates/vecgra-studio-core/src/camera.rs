use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub center: Vec2,
    pub zoom: f32,
}

pub const MIN_CAMERA_ZOOM: f32 = 0.08;
pub const MAX_CAMERA_ZOOM: f32 = 1_024.0;

impl Default for Camera {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

impl Camera {
    pub fn fit(bounds: Rect) -> Self {
        Self {
            center: bounds.center(),
            zoom: 1.0,
        }
    }

    /// Frames a subregion while retaining the full scene as the projection's
    /// scale reference. This lets search/focus navigation animate the camera
    /// without rebuilding world coordinates.
    pub fn framed(focus: Rect, world: Rect, viewport: Vec2, padding: f32) -> Self {
        let padding = padding.max(0.0);
        let available_width = (viewport.x - padding * 2.0).max(1.0);
        let available_height = (viewport.y - padding * 2.0).max(1.0);
        let desired_scale = (available_width / focus.width())
            .min(available_height / focus.height())
            .max(0.000_1);
        let base_scale = Self::fit(world).scale(world, viewport);
        Self {
            center: focus.center(),
            zoom: (desired_scale / base_scale).clamp(0.08, 24.0),
        }
    }

    pub fn scale(self, bounds: Rect, viewport: Vec2) -> f32 {
        let available_width = (viewport.x - 64.0).max(1.0);
        let available_height = (viewport.y - 64.0).max(1.0);
        let fitted = (available_width / bounds.width())
            .min(available_height / bounds.height())
            .max(0.000_1);
        fitted * self.zoom
    }

    pub fn project(self, point: Vec2, bounds: Rect, viewport: Vec2) -> Vec2 {
        let scale = self.scale(bounds, viewport);
        (point - self.center) * scale + viewport * 0.5
    }

    pub fn unproject(self, point: Vec2, bounds: Rect, viewport: Vec2) -> Vec2 {
        let scale = self.scale(bounds, viewport);
        (point - viewport * 0.5) / scale + self.center
    }

    pub fn pan_screen(&mut self, delta: Vec2, bounds: Rect, viewport: Vec2) {
        let scale = self.scale(bounds, viewport);
        self.center = self.center - delta / scale;
    }

    pub fn zoom_about(&mut self, screen_point: Vec2, factor: f32, bounds: Rect, viewport: Vec2) {
        let before = self.unproject(screen_point, bounds, viewport);
        self.zoom = (self.zoom * factor).clamp(MIN_CAMERA_ZOOM, MAX_CAMERA_ZOOM);
        let after = self.unproject(screen_point, bounds, viewport);
        self.center += before - after;
    }
}

/// Interruptible camera spring used by search and graph-focus navigation.
/// Zoom integrates in log space so magnification stays positive and feels
/// symmetric when entering or leaving a neighborhood.
#[derive(Clone, Copy, Debug)]
pub struct CameraMotion {
    target: Camera,
    center_velocity: Vec2,
    log_zoom_velocity: f32,
    moving: bool,
}

impl CameraMotion {
    pub fn new(camera: Camera) -> Self {
        Self {
            target: camera,
            center_velocity: Vec2::ZERO,
            log_zoom_velocity: 0.0,
            moving: false,
        }
    }

    pub fn retarget(&mut self, camera: Camera) {
        if !camera.center.x.is_finite()
            || !camera.center.y.is_finite()
            || !camera.zoom.is_finite()
            || camera.zoom <= 0.0
        {
            return;
        }
        self.target = camera;
        self.moving = true;
    }

    pub fn cancel_at(&mut self, camera: Camera) {
        self.target = camera;
        self.center_velocity = Vec2::ZERO;
        self.log_zoom_velocity = 0.0;
        self.moving = false;
    }

    pub const fn is_moving(&self) -> bool {
        self.moving
    }

    pub fn step(&mut self, camera: &mut Camera, elapsed_seconds: f32, reduce_motion: bool) -> bool {
        if !self.moving {
            return false;
        }
        if reduce_motion {
            *camera = self.target;
            self.cancel_at(*camera);
            return false;
        }
        let elapsed = elapsed_seconds.clamp(0.0, 0.05);
        if elapsed <= f32::EPSILON {
            return true;
        }
        const MAX_STEP: f32 = 1.0 / 120.0;
        const STIFFNESS: f32 = 145.0;
        const DAMPING: f32 = 24.0;
        let steps = (elapsed / MAX_STEP).ceil().max(1.0) as usize;
        let dt = elapsed / steps as f32;
        let target_log_zoom = self.target.zoom.max(0.001).ln();
        let mut log_zoom = camera.zoom.max(0.001).ln();
        for _ in 0..steps {
            let center_displacement = self.target.center - camera.center;
            let center_acceleration =
                center_displacement * STIFFNESS - self.center_velocity * DAMPING;
            self.center_velocity += center_acceleration * dt;
            camera.center += self.center_velocity * dt;

            let zoom_displacement = target_log_zoom - log_zoom;
            let zoom_acceleration =
                zoom_displacement * STIFFNESS - self.log_zoom_velocity * DAMPING;
            self.log_zoom_velocity += zoom_acceleration * dt;
            log_zoom += self.log_zoom_velocity * dt;
        }
        camera.zoom = log_zoom.exp().clamp(MIN_CAMERA_ZOOM, MAX_CAMERA_ZOOM);

        let center_distance = (self.target.center - camera.center).length_squared();
        let zoom_distance = (target_log_zoom - log_zoom).abs();
        if center_distance <= 0.000_4
            && self.center_velocity.length_squared() <= 0.002_5
            && zoom_distance <= 0.000_5
            && self.log_zoom_velocity.abs() <= 0.002
        {
            *camera = self.target;
            self.cancel_at(*camera);
            false
        } else if camera.center.x.is_finite()
            && camera.center.y.is_finite()
            && camera.zoom.is_finite()
        {
            true
        } else {
            *camera = self.target;
            self.cancel_at(*camera);
            false
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetailLevel {
    Overview,
    Communities,
    Elements,
}

pub fn detail_level(camera: Camera, node_count: usize) -> DetailLevel {
    if camera.zoom < 0.45 || (node_count >= 12_000 && camera.zoom < 1.6) {
        DetailLevel::Overview
    } else if camera.zoom < 2.8
        || (node_count > 25_000 && camera.zoom < 24.0)
        || (node_count > 12_000 && camera.zoom < 12.0)
    {
        DetailLevel::Communities
    } else {
        DetailLevel::Elements
    }
}

pub fn hit_test_node(
    snapshot: &SceneSnapshot,
    camera: Camera,
    viewport: Vec2,
    screen_point: Vec2,
    radius: f32,
) -> Option<usize> {
    hit_test_positions(
        &snapshot.nodes.positions,
        camera,
        snapshot.bounds,
        viewport,
        screen_point,
        radius,
    )
}

pub fn hit_test_positions(
    positions: &[Vec2],
    camera: Camera,
    world_bounds: Rect,
    viewport: Vec2,
    screen_point: Vec2,
    radius: f32,
) -> Option<usize> {
    let radius_squared = radius * radius;
    let scale = camera.scale(world_bounds, viewport);
    let project = |position: Vec2| (position - camera.center) * scale + viewport * 0.5;
    positions
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, position)| {
            let delta = project(position) - screen_point;
            let distance = delta.length_squared();
            (distance <= radius_squared).then_some((index, distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
}

pub fn hit_test_edges(
    positions: &[Vec2],
    edges: &SceneEdges,
    camera: Camera,
    world_bounds: Rect,
    viewport: Vec2,
    screen_point: Vec2,
    radius: f32,
) -> Option<usize> {
    let radius_squared = radius * radius;
    let scale = camera.scale(world_bounds, viewport);
    let project = |position: Vec2| (position - camera.center) * scale + viewport * 0.5;
    edges
        .sources
        .iter()
        .zip(&edges.targets)
        .enumerate()
        .filter_map(|(index, (&source, &target))| {
            let from = project(*positions.get(source as usize)?);
            let to = project(*positions.get(target as usize)?);
            if screen_point.x < from.x.min(to.x) - radius
                || screen_point.x > from.x.max(to.x) + radius
                || screen_point.y < from.y.min(to.y) - radius
                || screen_point.y > from.y.max(to.y) + radius
            {
                return None;
            }
            let distance = point_segment_distance_squared(screen_point, from, to);
            (distance <= radius_squared).then_some((index, distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
}

fn point_segment_distance_squared(point: Vec2, from: Vec2, to: Vec2) -> f32 {
    let segment = to - from;
    let length_squared = segment.length_squared();
    if length_squared <= f32::EPSILON {
        return (point - from).length_squared();
    }
    let from_to_point = point - from;
    let projection = from_to_point
        .x
        .mul_add(segment.x, from_to_point.y * segment.y)
        / length_squared;
    let nearest = from + segment * projection.clamp(0.0, 1.0);
    (point - nearest).length_squared()
}
