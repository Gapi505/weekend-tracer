use num_traits::Pow;
use crate::vectors::{Transform, Vec2, Vec3};
use image::Image;
use crate::camera::Camera;

pub mod image;
pub mod vectors;
mod ray;
mod camera;
mod objects;

#[cfg(test)]
mod tests;

fn main() {
    println!("Hello, world!");
    let mut camera = Camera::new(
        1980.,
        16./9.,
        Transform::new_at(vec3!(0.,2.,-3.)),
        90.);
    camera.transform.rotate_around_axis(Vec3::right(), 20.);
    println!("{:?}", camera);
    let mut img = Image::from_camera(&camera);
    camera.raytrace(&mut img);
    img.save().unwrap();
    img.open().unwrap()
}
