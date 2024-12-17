use crate::camera::Camera;
use crate::vectors::{Vec2, Vec3};
use crate::{vec2, vec3};
use num_traits::float::FloatCore;
use std::fs::File;
use std::io::Write;
use std::process::Command;

#[derive(Debug)]
pub struct Image {
    pub size: Vec2<usize>,
    data: Vec<Vec3<f32>>,
    pub accumulated_data: Vec<Vec<Vec3<f32>>>,
    aspect: f32,
    file: Option<File>,
}

impl Image {
    pub fn new(size: Vec2<usize>) -> Image {
        let data = vec![Vec3::zero(); size.x * size.y];
        let accumulated_data = vec![vec![Vec3::zero()]; size.x * size.y];
        let aspect = size.x as f32 / size.y as f32;
        Self {
            size,
            data,
            accumulated_data,
            aspect,
            file: None,
        }
    }

    pub fn from_camera(cam: &Camera) -> Image {
        let size = cam.res;
        let data = vec![Vec3::zero(); size.x * size.y];
        let accumulated_data = vec![vec![Vec3::zero()]; size.x * size.y];
        let aspect = cam.aspect;
        Self {
            size,
            data,
            accumulated_data,
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
    pub fn set_acc(&mut self, pos: Vec2<usize>, color: Vec3<f32>) {
        let index = pos.y * self.size.x + pos.x;
        self.accumulated_data[index].push(color);
    }

    pub fn accumulate(&mut self) {
        for i in 0..self.accumulated_data.len() {
            let mut sum = Vec3::zero();
            if self.accumulated_data[i].len() != 1 {
                for j in 1..self.accumulated_data[i].len() {
                    sum += self.accumulated_data[i][j];
                }
            }
            self.data[i] = sum / self.accumulated_data[i].len() as f32;
        }
    }
    pub fn update_buffer(&mut self, buffer: &mut Vec<u32>) {
        self.accumulate();
        self.gamma_correction();

        if buffer.len() != self.size.x * self.size.y {
            panic!("wrong size buffer")
        }
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                let index = y * self.size.x + x;
                buffer[index] = (255 << 24) |
                                ((self.data[index].x.to_u8() as u32) << 16) |
                                ((self.data[index].y.to_u8() as u32) << 8) |
                                (self.data[index].z.to_u8() as u32);
            }
        }

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
