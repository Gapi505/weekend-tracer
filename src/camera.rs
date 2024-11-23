use num_traits::real::Real;
use crate::{vec2, vec3};
use crate::image::Image;
use crate::objects::World;
use crate::ray::Ray;
use crate::vectors::{Vec2, Vec3, Transform};
#[derive(Debug, Copy, Clone)]
pub struct Camera {
    viewport_width: f32,
    pub aspect: f32,
    pub res: Vec2<usize>,
    pub transform: Transform,
    fov: Vec2<f32>,
    pub delta: Vec2<f32>,
}
impl Camera {
    pub fn new(viewport_width: f32, aspect: f32, transform: Transform, fov: f32) -> Camera {
        let res = vec2!(viewport_width as usize, (viewport_width/aspect) as usize);
        let fov = vec2!(fov, fov/aspect);
        let delta = vec2!(fov.x / res.x as f32, fov.y / res.y as f32);
        Camera{viewport_width, aspect, res, transform, fov, delta,..Self::default()}
    }

    pub fn spawn_ray_at_pixel(&self, pixel: Vec2<usize>) -> Ray {
        // Step 1: Center the pixel coordinates
        let centered_pixel = vec2!(
        pixel.x as f32 - (self.res.x as f32 / 2.0),
        pixel.y as f32 - (self.res.y as f32 / 2.0)
    );

        // Step 2: Convert pixel offsets to angular offsets in degrees
        let dir_deg = vec2!(
        centered_pixel.x * self.delta.x, // Horizontal angle (yaw)
        centered_pixel.y * self.delta.y  // Vertical angle (pitch)
    );

        // Step 3: Convert angles from degrees to radians
        let yaw = dir_deg.x.to_radians();
        let pitch = -dir_deg.y.to_radians(); // Invert pitch to account for screen coordinates

        // Step 4: Convert yaw and pitch to a 3D direction vector
        let mut dir = Vec3::new(
            yaw.sin() * pitch.cos(), // x component
            pitch.sin(),             // y component (up/down)
            pitch.cos() * yaw.cos()  // z component (forward)
        );

        // Step 5: Create and return the normalized ray
        Ray::new(self.transform.position, dir.normalize().rotate_around_origin(self.transform.rotation))
    }

    fn cast_ray(&self, ray: &mut Ray, world: &World) -> Vec3<f32>{
        ray.color = self.sky_color(ray.direction);

        let hit = world.collide(&ray);
        // print!("{}", hit.hit);
        if hit.hit{
            ray.color = (hit.normal + Vec3::new(1., 1., 1.))/2.;
        }
        ray.color
    }

    pub fn raytrace(&self, img: &mut Image){
        let print_progress = false;

        let mut world = World::new();
        world.default_scene();
        let total_scanlines = self.res.y;
        let step = (total_scanlines / 100).max(1); // Ensure we at least update once per step
        println!();
        for y in 0..self.res.y {
            for x in 0..self.res.x {
                let pixel = vec2!(x, y);
                let mut ray = self.spawn_ray_at_pixel(pixel);
                self.cast_ray(&mut ray, &world);
                img.setf(pixel, ray.color)
            }
            println!();
            // Report progress
            if y % step == 0 && print_progress {
                let progress = (y as f32 / total_scanlines as f32) * 100.0;
                print!("\x1b[A\x1b[2KProgress: {:.2}%\n", progress);
            }
        }
        if print_progress {
            print!("\x1b[A\x1b[2KProgress: {:.2}%\n", 100.);
        }
    }
    fn sky_color(&self, direction: Vec3<f32>) -> Vec3<f32> {
        // // Linear interpolation based on the y-component of the direction
        // let a = 0.5 * (direction.y + 1.0);
        // // Interpolate between white and sky blue
        // Vec3::new(1.0f32, 1.0, 1.0) * (1.0 - a) + Vec3::new(0.5f32, 0.7, 1.0) * a


        vec3!(direction.x, direction.y, direction.z)
    }
}
impl Default for Camera {
    fn default() -> Camera {
        Self{
            viewport_width: 256.,
            aspect: 1.,
            res: vec2![256, 256],
            transform: Transform::default(),
            fov: vec2!(90., 75.),
            delta: vec2!(0., 0.),
        }
    }
}

pub trait ToLocalSpace {
    fn to_local_space(&self, cam: &Camera) -> Self;
}