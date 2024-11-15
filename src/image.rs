use std::fs::File;
use std::io::Write;
use std::process::Command;


pub struct Image{
    width: u32,
    height: u32,
    data: Vec<[u8;3]>,
    file: Option<File>,
}
impl Image {
    pub fn hello_world() -> Self{
        let width = 256u32;
        let height = 256u32;
        let mut data = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                data.push([x as u8,y as u8,x as u8]);
            }
        }
        let file = None;
        Self{
            width,
            height,
            data,
            file
        }
    }
    pub fn save(&mut self) ->std::io::Result<()> {
        let mut file = File::create("image.ppm")?;
        write!(file, "P6\n{} {}\n{}\n", self.width, self.height, 255)?;
        let mut data = Vec::with_capacity(self.data.len()*3);
        for i in 0..self.data.len(){
            data.push(self.data[i][0]);
            data.push(self.data[i][1]);
            data.push(self.data[i][2]);
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
}
