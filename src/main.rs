use num_traits::ToPrimitive;
use crate::vectors::{Transform, Vec3};
use image::Image;
use crate::camera::Camera;
use crate::objects::World;



const RESOLUTION_X: usize = 720;




use crate::random::Random;
pub mod image;
pub mod vectors;
mod ray;
mod camera;

mod objects;
#[cfg(test)]
mod tests;
mod random;

fn main() {
    let start_time = std::time::Instant::now();
    let mut random =Random::new(69420);

    //init camera
    let mut camera = Camera::new_facing(RESOLUTION_X.to_f32().unwrap(),
                                        vec3!(0.0,1.0,-4.0), vec3!(0.0,0.0,5.),
    );
    println!("resolution: {}x{}px", camera.res.x, camera.res.y);
    println!("samples/px: {}, bounces: {}", camera.samples_per_pixel, camera.max_bounces);
    //camera.transform.rotate_around_axis(Vec3::right(), 20.);

    //create canvas
    let mut img = Image::from_camera(&camera);

    //create world
    let mut world = World::new();
    world.default_scene();
    world.build_bvtree();

    //render
    camera.raytrace(&mut img, &world, &mut random);
    let elapsed = start_time.elapsed().as_secs_f32();
    println!("Elapsed time: {}m {:.2}s",(elapsed/60.).floor(), elapsed%60.);

    //gama correct
    img.gamma_correction();

    //save
    img.save().unwrap();
    img.open().unwrap()
}
