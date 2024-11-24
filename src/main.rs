use crate::vectors::{Transform, Vec3};
use image::Image;
use crate::camera::Camera;
use crate::objects::World;
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
    let mut camera = Camera::new(
        1080.,
        16./9.,
        Transform::new_at(vec3!(0.,2.,-3.)),
        90.);
    println!("{}", camera.res);
    camera.transform.rotate_around_axis(Vec3::right(), 20.);

    //create canvas
    let mut img = Image::from_camera(&camera);

    //create world
    let mut world = World::new();
    world.default_scene();

    //render
    camera.raytrace(&mut img, &world, &mut random);
    let elapsed = start_time.elapsed().as_secs_f32();
    println!("Elapsed seconds: {:.2}s", elapsed);

    //gama correct
    img.gamma_correction();

    //save
    img.save().unwrap();
    img.open().unwrap()
}
