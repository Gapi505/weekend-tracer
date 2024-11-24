use crate::camera::{Camera, ToLocalSpace};
use crate::random::Random;
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
    pub(crate) material: Material,
}
impl HitRecord{
    pub fn new(hit: bool, t: f32, normal: Vec3<f32>, position: Vec3<f32>) -> HitRecord{
        HitRecord{hit, t, normal, position, front_face: false, ..Self::default()}
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
            material: Material::default()
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
    pub material: Material,
}
impl Object{
    pub fn new(transform: Transform, object_type: ObjectType, radius: f32) -> Object {
        Object{transform, object_type, radius, material: Material::default()}
    }
    pub fn new_sphere(transform: Transform, radius: f32) -> Object {
        Object{transform, object_type: ObjectType::Sphere, radius, material: Material::default()}
    }
    pub fn new_sphere_with_material(transform: Transform, radius: f32, material: Material) -> Object {
        Object{transform, object_type: ObjectType::Sphere, radius, material}
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
        if !ray.interval.surrounds(root) {
            root = (h + sqrt_d) / a;
            if !ray.interval.surrounds(root) {
                return hit;
            }
        }

        // Populate the hit record
        hit.hit = true;
        hit.t = root;
        hit.position = ray.at(hit.t);
        hit.material = self.material;

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
            material: Material::default()
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
        let mut sphere1 = Object::new_sphere(
            Transform::new_at(vec3!(0., 0., 5.)),
            1.);
        let floor = Object::new_sphere(
            Transform::new_at(vec3!(0., -6., 5.)),
            5.);
        let mut sphere2 = Object::new_sphere(
            Transform::new_at(vec3!(-1.25, -0.3, 4.4)),
            0.5
        );
        sphere1.material.metalness = 0.9;
        sphere1.material.roughness = 0.2;
        sphere1.material.albedo = vec3!(0.22, 0.753, 0.949);

        sphere2.material.albedo = vec3!(1., 0.2, 0.2);

        let mut sphere3 = Object::new_sphere(
            Transform::new_at(vec3!(2., 0.5, 5.)),
            0.9,
        );
        sphere3.material.emission = vec3!(2., 0.5, 0.5);

        let mut glass_ball = Object::new_sphere(
            Transform::new_at(vec3!(-0.6, 0.1, 3.)),
            0.6
        );
        glass_ball.material.albedo = Vec3::one();
        glass_ball.material.transmission = 1.;
        glass_ball.material.ior = 1.47;
        glass_ball.material.roughness = 0.0;


        let mut light2 = Object::new_sphere(
            Transform::new_at(vec3!(-3., 1., 3.)),
            0.3
        );
        light2.material.emission = vec3!(7.,7., 7.);

        self.add(sphere1);
        self.add(sphere2);
        self.add(floor);
        self.add(sphere3);
        self.add(glass_ball);
        self.add(light2);
    }

    pub fn simple_scene(&mut self){
        let sphere1 = Object::new_sphere_with_material(
            Transform::new_at(vec3!(0., 0., 5.)),
            1.,
            Material::new_diffuse(Vec3::new(1., 1., 1.)),
        );
        let sphere2 = Object::new_sphere_with_material(
            Transform::new_at(vec3!(0., -11., 5.)),
            10.,
            Material::new_diffuse(Vec3::new(1., 1., 1.)),
        );
        self.add(sphere1);
        self.add(sphere2);
    }

    pub fn collide(&self, ray: &Ray) -> HitRecord{
        let mut closest_hit = HitRecord::default();
        closest_hit.t = ray.interval.max;
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


#[derive(Debug, Copy, Clone)]
pub struct Material{
    pub(crate) albedo: Vec3::<f32>,
    metalness: f32,
    roughness: f32,
    ior: f32,
    pub(crate) emission: Vec3::<f32>,
    transmission: f32,

}
impl Material{
    pub fn new_diffuse(albedo: Vec3<f32>) -> Self{
        Self{
            albedo,
            ..Self::default()
        }
    }
    pub(crate) fn scatter(&self, normal: Vec3<f32>, direction: Vec3<f32>, rng: &mut Random) -> Vec3<f32> {
        let out = rng.random_unit_vector() + normal;
        out
    }

    // Schlick approximation for Fresnel effect
    fn schlick(&self, cosine: f32, ior: f32) -> f32 {
        let r0 = ((1.0 - ior) / (1.0 + ior)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Default for Material {
    fn default() -> Material {
        Self{
            albedo: Vec3::one(),
            metalness: 0.,
            roughness: 1.,
            ior: 1.,
            emission: Vec3::zero(),
            transmission: 0.,
        }
    }
}