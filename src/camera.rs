use num_traits::real::Real;
use crate::{vec2, vec3};
use crate::image::Image;
use crate::objects::World;
use crate::random::Random;
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
    pub samples_per_pixel: usize,
    pub max_bounces: usize,
    pub gamut: f32,
}
impl Camera {
    pub fn new(viewport_width: f32, aspect: f32, transform: Transform, fov: f32) -> Camera {
        let res = vec2!(viewport_width as usize, (viewport_width/aspect) as usize);
        let fov = vec2!(fov, fov/aspect);
        let delta = vec2!(fov.x / res.x as f32, fov.y / res.y as f32);
        Camera{viewport_width, aspect, res, transform, fov, delta,..Self::default()}
    }

    pub fn spawn_ray_at_pixel(&self, pixel: Vec2<usize>, rng: &mut Random) -> Ray {
        // Step 1: Center the pixel coordinates
        let mut centered_pixel = vec2!(
            pixel.x as f32 - (self.res.x as f32 / 2.0),
            pixel.y as f32 - (self.res.y as f32 / 2.0)
        );
        centered_pixel = rng.sample_square() + centered_pixel;

        // Step 2: Convert pixel offsets to angular offsets in degrees
        let dir_deg = vec2!(
        (centered_pixel.x) * self.delta.x, // Horizontal angle (yaw)
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

    fn cast_ray(&self, ray: &mut Ray, world: &World, depth: usize, rng: &mut Random) -> Vec3<f32>{
        if depth > self.max_bounces {
            return vec3!(0.0, 0.0, 0.0);
        }
        let hit = world.collide(&ray);
        // print!("{}", hit.hit);
        if hit.hit{
            // let dir = (hit.normal + rng.random_unit_vector()).normalize();
            let (dir, attenuation) = hit.material.scatter(hit.normal, ray.direction, rng, hit.front_face);
            // let dir = hit.normal;
            let reflected_color= self.cast_ray(
                &mut Ray::new(hit.position, dir),
                world,
                depth + 1,
                rng
            );
            // let albedo = hit.material.albedo;
            // let emission = hit.material.emission;
            // ray.color = (reflected_color * albedo) + emission;
            ray.color = (reflected_color * attenuation) + hit.material.emission * hit.material.emission_strength;
            ray.color *= self.gamut;
            return ray.color;
        }

        ray.color = self.sky_color(ray.direction);
        ray.color
    }

    pub fn raytrace(&self, img: &mut Image, world: &World, rng: &mut Random) {
        let print_progress = true;
        let total_scanlines = self.res.y;
        let step = (total_scanlines / 1080).max(1); // Ensure we at least update once per step
        let pixel_sample_influence = 1. / self.samples_per_pixel as f32;
        println!();
        for y in 0..self.res.y {
            for x in 0..self.res.x {
                let pixel = vec2!(x, y);
                let mut pixel_color = Vec3::zero();
                for _ in 0..self.samples_per_pixel {
                    let mut ray = self.spawn_ray_at_pixel(pixel, rng);
                    self.cast_ray(&mut ray, &world, 0, rng);
                    pixel_color +=  ray.color;
                }
                pixel_color *= pixel_sample_influence;
                img.set(pixel, pixel_color);
            }
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
        // Linear interpolation based on the y-component of the direction
        let a = 0.5 * (direction.y + 1.0);
        // Interpolate between white and sky blue
        Vec3::new(1.0f32, 1.0, 1.0) * (1.0 - a) + Vec3::new(0.5f32, 0.7, 1.0) * a


        // vec3!(direction.x, direction.y, direction.z)
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
            samples_per_pixel: 500,
            max_bounces: 20,
            gamut: 0.5,
        }
    }
}

pub trait ToLocalSpace {
    fn to_local_space(&self, cam: &Camera) -> Self;
}