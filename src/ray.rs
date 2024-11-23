use crate::vec3;
use crate::vectors::Vec3;

pub(crate) struct Ray{
    pub(crate) origin: Vec3<f32>,
    pub(crate) direction: Vec3<f32>,
    pub color: Vec3<f32>,
    pub min: f32,
    pub max: f32,
}
impl Ray{
    pub(crate) fn new(origin: Vec3<f32>, direction: Vec3<f32>) -> Self{
        Self{origin, direction, color: Vec3::<f32>::zero(), ..Self::default()}
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
            min: 0.,
            max: f32::INFINITY,
        }
    }
}

