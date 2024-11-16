use crate::vectors::{Vec2, Vec3};
use crate::{vec2, vec3};
use std::fs::File;
use std::io::Write;
use std::process::Command;

pub struct Image{
    size:Vec2<usize>,
    data: Vec<Vec3<u8>>,
    aspect:f32,
    file: Option<File>,
}
impl Image {
    pub fn new(size: Vec2<usize>) -> Image {
        let mut data = vec![Vec3::zero();size.x * size.y];
        let aspect = size.x as f32 / size.y as f32;
        Self{
            size,
            data,
            aspect,
            file: None
        }
    }
    pub fn hello_world(&mut self){
        for y in 0..self.size.y{
            for x in 0..self.size.x{
                self.set(vec2!(x,y), vec3!(0, 0, 0));
            }
        }
    }
    pub fn save(&mut self) ->std::io::Result<()> {
        let mut file = File::create("image.ppm")?;
        write!(file, "P6\n{} {}\n{}\n", self.size.y, self.size.y, 255)?;
        let mut data = Vec::with_capacity(self.data.len()*3);
        for i in 0..self.data.len(){
            data.push(self.data[i].x);
            data.push(self.data[i].y);
            data.push(self.data[i].z);
        }
        file.write_all(&data)?;
        self.file = Some(file);
        Ok(())
    }
    pub fn open(&mut self) -> Result<(), Box<dyn std::error::Error>>{
        if self.file.is_some(){
            Command::new("xdg-open")
                .arg("image.ppm")
                .spawn()?
                .wait()?;
        }
        Ok(())
    }
    pub fn set(&mut self, pos: Vec2<usize>, color: Vec3<u8>) {
        let index = pos.y * self.size.y + pos.x;
        self.data[index] = color;
    }
}
