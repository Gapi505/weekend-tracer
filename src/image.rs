use crate::vectors::{Vec2, Vec3};
use crate::{camera, vec2, vec3};
use std::fs::File;
use std::io::Write;
use std::process::Command;
use num_traits::float::FloatCore;
use crate::camera::Camera;

pub struct Image {
    size: Vec2<usize>,
    data: Vec<Vec3<f32>>,
    aspect: f32,
    file: Option<File>,
}

impl Image {
    pub fn new(size: Vec2<usize>) -> Image {
        let data = vec![Vec3::zero(); size.x * size.y];
        let aspect = size.x as f32 / size.y as f32;
        Self {
            size,
            data,
            aspect,
            file: None,
        }
    }

    pub fn from_camera(cam: &Camera) -> Image {
        let size = cam.res;
        let data = vec![Vec3::zero(); size.x * size.y];
        let aspect = cam.aspect;
        Self {
            size,
            data,
            aspect,
            file: None,
        }
    }

    pub fn hello_world(&mut self) {
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                self.set(vec2!(x, y), vec3!(0.0, 0.0, 0.0));
            }
        }
    }

    pub fn gamma_correction(&mut self) {
        for c in self.data.iter_mut() {
            c.x.linear_to_gamma();
            c.y.linear_to_gamma();
            c.z.linear_to_gamma();
        }
    }

    pub fn save(&mut self) -> std::io::Result<()> {
        let mut file = File::create("image.ppm")?;
        write!(file, "P6\n{} {}\n{}\n", self.size.x, self.size.y, 255)?;
        let mut data = Vec::with_capacity(self.data.len() * 3);
        for color in &self.data {
            data.push(color.x.to_u8());
            data.push(color.y.to_u8());
            data.push(color.z.to_u8());
        }
        file.write_all(&data)?;
        self.file = Some(file);
        Ok(())
    }

    pub fn open(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.file.is_some() {
            Command::new("ffmpeg")
                .args(["-y", "-i", "image.ppm", "image.png"])
                .stdout(std::process::Stdio::null()) // Suppress stdout
                .stderr(std::process::Stdio::null())
                .spawn()?
                .wait()?;
            Command::new("xdg-open")
                .arg("image.png")
                .spawn()?
                .wait()?;
        }
        Ok(())
    }

    pub fn set(&mut self, pos: Vec2<usize>, color: Vec3<f32>) {
        let index = pos.y * self.size.x + pos.x;
        self.data[index] = color;
    }
}

trait FToU8Color {
    fn to_u8(&self) -> u8;
}

impl FToU8Color for f32 {
    fn to_u8(&self) -> u8 {
        (self.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    }
}

trait LinearToGamma {
    fn linear_to_gamma(&mut self);
}

impl LinearToGamma for f32 {
    fn linear_to_gamma(&mut self) {
        if *self > 0.0 {
            *self = self.sqrt();
        } else {
            *self = 0.0;
        }
    }
}
