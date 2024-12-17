use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;
use std::thread::sleep;
use std::time::Duration;
use num_traits::ToPrimitive;
use crate::vectors::{Transform, Vec3};
use image::Image;
use crate::camera::Camera;
use crate::objects::World;
use minifb::{Key, Window, WindowOptions};
use log::log;

const RESOLUTION_X: usize = 720;
const TRACING_MODE: usize = 1;



use crate::random::Random;
pub mod image;
pub mod vectors;
mod ray;
mod camera;

mod objects;
#[cfg(test)]
mod tests;
mod random;

#[tokio::main]
async fn main() {
    let start_time = std::time::Instant::now();
    let mut random =Random::new(69420);

    //init camera
    let mut camera = Camera::new_facing(RESOLUTION_X.to_f32().unwrap(),
                                        vec3!(2.,1.0,-4.0), vec3!(0.0,0.0,5.),
    );
    println!("resolution: {}x{}px", camera.res.x, camera.res.y);
    if TRACING_MODE ==0{
        println!("samples/px: {}, bounces: {}", camera.samples_per_pixel, camera.max_bounces);

    }
    else if TRACING_MODE ==1 {
        println!("bounces: {}", camera.max_bounces);
    }
    //camera.transform.rotate_around_axis(Vec3::right(), 20.);

    //create canvas
    let mut img = Image::from_camera(&camera);

    let mut window_buffer = vec![0; img.size.x *img.size.y];



    //create world
    let mut world = World::new();
    world.default_scene();
    world.build_bvtree();

    let mut img_arc = Arc::new(Mutex::new(img));

    // render
    match TRACING_MODE {
        0=>{
            camera.full_raytrace(&mut img_arc.lock().unwrap(), &world, &mut random);
        }
        1 =>{
            let res = camera.res.clone();
            let mut window = Window::new(
                "raytracer",
                res.x,
                res.y,
                WindowOptions::default(),
            ).unwrap_or_else(|e| panic!("{}", e));
            window.update_with_buffer(&window_buffer, res.x, res.y).unwrap();
            env_logger::builder()
                .filter_module("winit", log::LevelFilter::Warn)
                .init();
            let mut img_arc_clone = img_arc.clone();
            let keep_tracing = camera.keep_tracing.clone();
            let tracer_async = tokio::spawn(async move {
                camera.random_sample_raytrace(&mut img_arc_clone, &world, &mut random).await;

            });

            // Main loop
            let mut i = 0;
            while window.is_open() && !window.is_key_down(Key::Escape) && keep_tracing.load(Ordering::Relaxed) {
                window.update_with_buffer(&window_buffer, res.x, res.y).unwrap();
                if i >= res.x/10{
                    img_arc.lock().unwrap().update_buffer(&mut window_buffer);
                    i = 0;
                }
                i += 1
            }
            //sleep(Duration::from_millis(100));

            // Stop tracing
            keep_tracing.store(false, std::sync::atomic::Ordering::Relaxed);
            let elapsed = start_time.elapsed().as_secs_f32();
            println!("Elapsed time: {}m {:.2}s",(elapsed/60.).floor(), elapsed%60.);

            // Await the async task
            tracer_async.await.unwrap();
            img_arc.lock().unwrap().accumulate();
            img_arc.lock().unwrap().accumulated_data_denoise();
            img_arc.lock().unwrap().update_data();

        }
        _ => {

        }

    }



    //gama correct
    img_arc.lock().unwrap().gamma_correction();

    //save
    img_arc.lock().unwrap().save().unwrap();
    img_arc.lock().unwrap().open().unwrap();
}
