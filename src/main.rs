pub mod image;
pub mod vectors;

fn main() {
    println!("Hello, world!");
    let mut img = image::Image::hello_world();
    img.save().unwrap();
    img.open().unwrap()
}
