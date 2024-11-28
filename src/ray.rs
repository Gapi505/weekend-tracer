use crate::vec3;
use crate::vectors::Vec3;

pub(crate) struct Ray{
    pub(crate) origin: Vec3<f32>,
    pub(crate) direction: Vec3<f32>,
    pub color: Vec3<f32>,
    pub interval: Interval,
    pub time: f32,
}
impl Ray{
    pub(crate) fn new(origin: Vec3<f32>, direction: Vec3<f32>) -> Self{
        Self{origin, direction, color: Vec3::<f32>::zero(), ..Self::default()}
    }
    pub fn new_at_time(origin: Vec3<f32>, direction: Vec3<f32>, time: f32) -> Self{
        Self{origin, direction, time,..Self::default()}
    }
    pub(crate) fn at(&self, t: f32) -> Vec3<f32>{
        self.origin + self.direction * t
    }
}

impl Default for Ray{
    fn default()->Self{
        Self{
            origin: Vec3::zero(),
            direction: vec3![0., 0., 1.],
            color: Vec3::zero(),
            interval: Interval::new(0.00001, f32::INFINITY),
            time: 0.,
        }
    }
}


pub struct Interval{
    pub(crate) min: f32,
    pub(crate) max: f32,
}
impl Interval{
    pub fn new(min: f32, max: f32) -> Self{
        Self{min, max}
    }
    pub fn zero_to(to: f32) -> Self{
        Self{min:0., max:to}
    }
    pub fn size(&self) -> f32{
        self.max - self.min
    }
    pub fn contains(&self, t:f32) -> bool {
        self.min <= t && t <= self.max
    }
    pub fn surrounds(&self, t:f32) -> bool{
        self.min < t && t < self.max
    }
    pub fn empty() -> Self{
        Self{min:f32::INFINITY, max:-f32::INFINITY}
    }
    pub fn universe() -> Self{
        Self{min:-f32::INFINITY, max:f32::INFINITY}
    }
    pub fn clamp(&self, x:f32)->f32{
        if x<self.min{return self.min};
        if x>self.max{return self.max};
        x
    }
}

