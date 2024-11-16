use crate::vectors::Vec2;
use image::Image;
pub mod image;
pub mod vectors;


fn main() {
    println!("Hello, world!");
    let mut img = Image::new(vec2!(256,256));
    img.hello_world();
    img.save().unwrap();
    img.open().unwrap()
}
