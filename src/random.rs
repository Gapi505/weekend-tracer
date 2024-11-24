use rand::{Rng, SeedableRng};
use rand::distr::Uniform;
use rand_chacha::ChaCha8Rng;
use crate::vec2;
use crate::vectors::{Vec2, Vec3};

pub struct Random {
    seed: u64,
    rng:ChaCha8Rng
}
impl Random {
    pub fn new(seed: u64) -> Random {
        let rng = ChaCha8Rng::seed_from_u64(seed);
        Self{seed, rng}
    }
    pub fn randf(&mut self) -> f32 {
        self.rng.random::<f32>()
    }
    pub fn randr(&mut self, min: f32, max: f32) -> f32 {
        self.rng.gen_range(min..max)
    }
    pub fn sample_square(&mut self)  -> Vec2<f32> {
        vec2!(self.rng.random::<f32>(), self.rng.random::<f32>())
    }

    pub fn random_unit_vector(&mut self) -> Vec3<f32>{
        loop{
            let p = Vec3::new(
                self.randr(-1.0, 1.0),
                self.randr(-1.0, 1.0),
                self.randr(-1.0, 1.0),
            );
            let lensq = p.length_sq();
            if lensq < 1.0 && lensq > f32::EPSILON { return p/lensq.sqrt(); }
        }
    }
    pub fn random_in_unit_sphere(&mut self) -> Vec3<f32>{
        loop{
            let p = Vec3::new(
                self.randr(-1.0, 1.0),
                self.randr(-1.0, 1.0),
                self.randr(-1.0, 1.0),
            );
            let lensq = p.length_sq();
            if lensq < 1.0 && lensq > f32::MIN_POSITIVE { return p}
        }
    }

    pub fn random_on_hemisphere(&mut self, normal: Vec3<f32>) -> Vec3<f32> {
        let on_sphere = self.random_unit_vector();
        if on_sphere.dot(normal) > 0.0 {
            on_sphere
        }
        else {
            -on_sphere
        }
    }
}