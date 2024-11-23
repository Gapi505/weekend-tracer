use num_traits::Pow;
use crate::vectors::{Transform, Vec2, Vec3};
use image::Image;
use crate::camera::Camera;

pub mod image;
pub mod vectors;
mod ray;
mod camera;
mod objects;

fn main() {
    println!("Hello, world!");
    let mut camera = Camera::new(
        1980.,
        16./9.,
        Transform::new_at(vec3!(0.,2.,7.)),
        90.);
    camera.transform.rotate_by(vec3!(30.,180.,0.));
    println!("{:?}", camera);
    let mut img = Image::from_camera(&camera);
    camera.raytrace(&mut img);
    img.save().unwrap();
    img.open().unwrap()
}
