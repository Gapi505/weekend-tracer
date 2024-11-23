use crate::camera::{Camera, ToLocalSpace};
use crate::vectors::{Transform, Vec3};
use crate::ray::Ray;
use crate::vec3;


#[derive(Debug)]
pub struct HitRecord{
    pub hit : bool,
    t: f32,
    pub normal: Vec3<f32>,
    pub position: Vec3<f32>,
    front_face: bool,
}
impl HitRecord{
    pub fn new(hit: bool, t: f32, normal: Vec3<f32>, position: Vec3<f32>) -> HitRecord{
        HitRecord{hit, t, normal, position, front_face: false}
    }
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3<f32>){
        self.front_face = ray.direction.dot(outward_normal) < 0.0;
        self.normal = if self.front_face{ outward_normal } else { -outward_normal }
    }
}
impl Default for HitRecord {
    fn default() -> HitRecord{
        HitRecord{
            hit: false,
            t: 0.,
            normal: Vec3::<f32>::zero(),
            position: Vec3::<f32>::zero(),
            front_face: false,
        }
    }
}

pub enum ObjectType{
    Sphere
}
pub struct Object{
    transform: Transform,
    object_type: ObjectType,
    radius: f32,
}
impl Object{
    pub fn new(transform: Transform, object_type: ObjectType, radius: f32) -> Object {
        Object{transform, object_type, radius}
    }
    pub fn new_sphere(transform: Transform, radius: f32) -> Object {
        Object{transform, object_type: ObjectType::Sphere, radius}
    }
    fn collide_sphere(&self, ray: &Ray) -> HitRecord {
        let mut hit = HitRecord::default();
        // Initialize the hit record
        let oc = self.transform.position - ray.origin;
        let a = ray.direction.length_sq();

        // Prevent division by zero
        if a.abs() < f32::EPSILON {
            return hit;
        }

        let h = ray.direction.dot(oc);
        let c = oc.length_sq() - self.radius * self.radius;

        let discriminant = h * h - a * c;


        // Check for no hit
        if discriminant < 0.0 {
            return hit;
        }

        let sqrt_d = discriminant.sqrt();

        // Find the nearest root
        let mut root = (h - sqrt_d) / a;
        if root <= ray.min || root >= ray.max {
            root = (h + sqrt_d) / a;
            if root <= ray.min || root >= ray.max {
                return hit;
            }
        }

        // Populate the hit record
        hit.hit = true;
        hit.t = root;
        hit.position = ray.at(hit.t);

        let outward_normal = (hit.position - self.transform.position) / self.radius;
        hit.set_face_normal(&ray, outward_normal);

        hit
    }
    pub(crate) fn collide(&self, ray: &Ray) -> HitRecord{
        match self.object_type {
            ObjectType::Sphere => {self.collide_sphere(ray)},
        }
    }
}

#[test]
fn normal_center(){
    let mut ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0., 1.));
    let sphere = Object::new_sphere(Transform::new_at(vec3!(0.,0., 5.)), 1.);
    let hit = sphere.collide(&mut ray);
    assert_eq!(hit.normal, vec3!(0.,0.,-1.));
}


impl Default for Object{
    fn default() -> Self {
        Self{
            transform: Transform::new_at(Vec3::zero()),
            object_type: ObjectType::Sphere,
            radius: 1.0,
        }
    }
}


pub struct World{
    pub(crate) objects: Vec<Object>,
}
impl World {
    pub fn new() -> World {
        World { objects: vec![] }
    }
    pub fn add(&mut self, object: Object) {
        self.objects.push(object);
    }
    pub fn default_scene(&mut self){
        self.add(Object::new_sphere(
            Transform::new_at(vec3!(0., 0., 3.)),
            1.));
        self.add(Object::new_sphere(
            Transform::new_at(vec3!(0., -6., 5.)),
            5.));

    }

    pub fn collide(&self, ray: &Ray) -> HitRecord{
        let mut closest_hit = HitRecord::default();
        closest_hit.t = ray.max;
        for object in &self.objects{
            let hit = object.collide(ray);
            if !hit.hit{
                continue
            }
            // println!("hit: {:?}", &hit);
            if hit.t < closest_hit.t {
                closest_hit = hit;
                //print!("{:?}", closest_hit)
            }
        }
        closest_hit
    }
}