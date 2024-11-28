use num_traits::ToPrimitive;
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
    pub(crate) front_face: bool,
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
    Sphere,
    MovingSphere,
}
pub struct Object{
    transform: Transform,
    moved_transform: Transform,
    object_type: ObjectType,
    radius: f32,
    pub material: Material,
}
impl Object{
    pub fn new(transform: Transform, object_type: ObjectType, radius: f32) -> Object {
        Object{transform, object_type, radius, material: Material::default(), moved_transform: transform}
    }
    pub fn new_sphere(transform: Transform, radius: f32) -> Object {
        Object{transform, object_type: ObjectType::Sphere, radius, material: Material::default(), moved_transform: transform}
    }
    pub fn new_sphere_with_material(transform: Transform, radius: f32, material: Material) -> Object {
        Object{transform, object_type: ObjectType::Sphere, radius, material, moved_transform: transform}
    }
    pub fn new_moving_sphere_with_material(transform: Transform, moved_transform: Transform, radius: f32, material: Material) -> Object {
        Object{
            transform,
            moved_transform,
            radius,
            material,
            object_type: ObjectType::MovingSphere,
        }
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

    fn collide_moving_sphere(&self, ray: &Ray) -> HitRecord{
        self.collide_sphere(ray)
    }
    pub(crate) fn collide(&self, ray: &Ray) -> HitRecord{
        match self.object_type {
            ObjectType::Sphere => {self.collide_sphere(ray)},
            ObjectType::MovingSphere => {self.collide_moving_sphere(ray)}
        }
    }
}

#[test]
fn normal_center(){
    let mut ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0., 1.));
    let sphere = Object::new_sphere(Transform::at(vec3!(0.,0., 5.)), 1.);
    let hit = sphere.collide(&mut ray);
    assert_eq!(hit.normal, vec3!(0.,0.,-1.));
}


impl Default for Object{
    fn default() -> Self {
        Self{
            transform: Transform::at(Vec3::zero()),
            object_type: ObjectType::Sphere,
            radius: 1.0,
            material: Material::default(),
            moved_transform: Transform::at(Vec3::zero()),
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
    pub fn default_scene(&mut self) {
        let sphere1 = Object::new_sphere_with_material(
            Transform::at(vec3!(0., 0., 5.)),
            1.,
            Material::new(
                vec3!(0.2, 0.7, 0.9), // albedo
                0.9,                       // metalness
                0.4,                       // roughness
                0.0,                       // transmission
            ),
        );

        let floor = Object::new_sphere_with_material(
            Transform::at(vec3!(0., -6., 5.)),
            5.,
            Material::new_diffuse(Vec3::one()),
        );

        let sphere2 = Object::new_sphere_with_material(
            Transform::at(vec3!(-1.25, -0.3, 4.4)),
            0.5,
            Material::new_diffuse(vec3!(1., 0.1, 0.1)),
        );

        let sphere3 = Object::new_sphere_with_material(
            Transform::at(vec3!(3., 2., 5.)),
            0.9,
            Material::new(
                vec3!(1., 0.2, 0.2),
                0.9,
                0.05,
                0.
            ),
        );
        let sphere4 = Object::new_sphere_with_material(
            Transform::at(vec3!(1., 2., 5.)),
            0.7,
            Material::new(
                vec3!(0.3, 1., 0.1),
                1.,
                0.,
                0.,
            )
        );

        let glass_ball = Object::new_sphere_with_material(
            Transform::at(vec3!(-0.6, 0.1, 3.)),
            0.6,
            Material::new(
                Vec3::one(), // albedo
                0.0,         // metalness
                0.001,         // roughness
                1.0,         // transmission
            ).with_ior(1.47),
        );

        let another_ball = Object::new_sphere_with_material(
            Transform::at(vec3!(-3., 1., 4.)),
            1.,
            Material::new_diffuse(vec3!(1., 1., 1.)).with_emission(vec3!(1.,1.,1.), 1.),
        );
        let close_sphere = Object::new_sphere_with_material(
            Transform::at(vec3!(1., 1., 3.)),
            0.5,
            Material::new_diffuse(vec3!(0.2, 0.8, 0.8)),
        );

        let big_mirror = Object::new_sphere_with_material(
            Transform::at(vec3!(-8., -2., 11.5)),
            8.,
            Material::new(
                vec3!(0.9, 0.9, 0.9),
                1.,
                0.0,
                0.
            ),
        );

        for z in 1..10{
            let depth_ball = Object::new_sphere_with_material(
                Transform::at(vec3!(3., -1., z as f32)),
                0.5,
                Material::new_diffuse(vec3!(0.2, 0.8, 0.8)),
            );
            self.add(depth_ball);
        }

        self.add(sphere1);
        self.add(sphere2);
        self.add(sphere3);
        self.add(sphere4);
        self.add(floor);
        self.add(glass_ball);
        self.add(another_ball);
        self.add(close_sphere);
        self.add(big_mirror);
    }

    pub fn simple_scene(&mut self){
        let sphere1 = Object::new_sphere_with_material(
            Transform::at(vec3!(0., 0., 5.)),
            1.,
            Material::new_diffuse(Vec3::new(1., 1., 1.)),
        );
        let sphere2 = Object::new_sphere_with_material(
            Transform::at(vec3!(0., -11., 5.)),
            10.,
            Material::new_diffuse(Vec3::new(1., 1., 1.)).with_roughness(0.1),
        );
        self.add(sphere1);
        self.add(sphere2);
    }

    pub fn refraction_test(&mut self){
        let ground = Object::new_sphere_with_material(
            Transform::at(vec3!(0., -21., 5.)),
            20.,
            Material::new_diffuse(Vec3::new(1., 1., 1.)),
        );
        let glass_ball = Object::new_sphere_with_material(
            Transform::at(vec3!(0., 0., 5.)),
            1.,
            Material::new_translucent(1., 0., 1.37)
        );
        self.add(glass_ball);
        self.add(ground);
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
    pub albedo: Vec3::<f32>,
    metalness: f32,
    roughness: f32,
    ior: f32,
    pub emission: Vec3::<f32>,
    pub emission_strength: f32,
    transmission: f32,

}
impl Material{
    pub fn new_diffuse(albedo: Vec3<f32>) -> Self{
        Self{
            albedo,
            ..Self::default()
        }
    }
    pub fn new_light(emission: Vec3<f32>, emission_strength: f32) -> Self{
        Self{
            emission,
            emission_strength,
            ..Self::default()
        }
    }
    pub fn new_translucent(transmission: f32, roughness: f32, ior: f32) -> Self{
        Self{
            transmission,
            roughness,
            ior,
            ..Self::default()
        }
    }
    pub fn new(albedo: Vec3<f32>, metalness: f32, roughness: f32, transmission: f32) -> Self{
        Self{
            albedo,
            metalness,
            roughness,
            transmission,
            ..Self::default()
        }
    }

    pub fn with_ior(&mut self, ior: f32) -> Self{
        self.ior = ior;
        *self
    }
    pub fn with_roughness(&mut self, roughness: f32) -> Self{
        self.roughness = roughness;
        *self
    }
    pub fn with_emission(&mut self, emission: Vec3<f32>, emission_strength: f32) -> Self{
        self.emission = emission;
        self.emission_strength = emission_strength;
        *self
    }
    pub fn with_albedo(&mut self, albedo: Vec3<f32>) -> Self{
        self.albedo = albedo;
        *self
    }
    pub fn with_metalness(&mut self, metalness: f32) -> Self{
        self.metalness = metalness;
        *self
    }
    pub fn with_transmission(&mut self, transmission: f32, ior: f32) -> Self{
        self.transmission = transmission;
        self.ior = ior;
        *self
    }
    pub(crate) fn scatter(
        &self,
        normal: Vec3<f32>,
        direction: Vec3<f32>,
        rng: &mut Random,
        is_front_face: bool,
    ) -> (Vec3<f32>, Vec3<f32>) {
        let unit_direction = direction.normalize();
        // let is_front_face = unit_direction.dot(normal) < 0.0;
        // let normal = if is_front_face { normal } else { -normal };

        // Handle emission
        if self.emission_strength > 0.0 {
            let emitted = self.emission * self.emission_strength;
            // Emission doesn't scatter further; return early
            return (Vec3::zero(), emitted);
        }

        // Clamp material properties once to avoid redundant operations
        let metalness = self.metalness.clamp(0.0, 1.0);
        let roughness = self.roughness.clamp(0.0, 1.0);
        let transmission = self.transmission.clamp(0.0, 1.0);

        // Early return for pure diffuse materials
        if metalness == 0.0 && transmission == 0.0 {
            let scattered_direction = normal + rng.random_unit_vector() * roughness;
            let attenuation = self.albedo;
            return (scattered_direction.normalize(), attenuation);
        }

        let mut attenuation = self.albedo;
        let mut scattered_direction = Vec3::zero();

        // Compute Fresnel reflectance using Schlick's approximation
        let cos_theta = (-unit_direction).dot(normal).min(1.0);
        let mut reflectance = self.schlick(cos_theta, self.ior);

        // Adjust reflectance based on metalness
        reflectance = reflectance * (1.0 - metalness) + metalness;

        // Skip refraction calculations if transmission is zero
        if transmission == 0.0 {
            // Only consider reflection and diffuse scattering
            let diffuse_prob = (1.0 - reflectance) * (1.0 - metalness);
            let specular_prob = reflectance;

            // Normalize probabilities
            let sum = diffuse_prob + specular_prob;
            let diffuse_prob = diffuse_prob / sum;
            let specular_prob = specular_prob / sum;

            let random_choice = rng.randf();

            if random_choice < diffuse_prob {
                // Diffuse reflection (Lambertian)
                scattered_direction = normal + rng.random_unit_vector() * roughness;
                attenuation = self.albedo;
            } else {
                // Specular reflection
                let reflected = reflect(unit_direction, normal);
                scattered_direction = reflected + roughness * rng.random_in_unit_sphere();
                attenuation = if metalness > 0.0 {
                    self.albedo
                } else {
                    Vec3::one()
                };
            }
        } else {
            // Transmission is non-zero; include refraction calculations
            let diffuse_prob = (1.0 - reflectance) * (1.0 - metalness) * (1.0 - transmission);
            let specular_prob = reflectance * (1.0 - transmission);
            let transmission_prob = transmission;

            // Normalize probabilities
            let sum = diffuse_prob + specular_prob + transmission_prob;
            let diffuse_prob = diffuse_prob / sum;
            let specular_prob = specular_prob / sum;
            let transmission_prob = transmission_prob / sum;

            let random_choice = rng.randf();

            if random_choice < diffuse_prob {
                // Diffuse reflection
                scattered_direction = normal + rng.random_unit_vector() * roughness;
                attenuation = self.albedo;
            } else if random_choice < diffuse_prob + specular_prob {
                // Specular reflection
                let reflected = reflect(unit_direction, normal);
                scattered_direction = reflected + roughness * rng.random_in_unit_sphere();
                attenuation = if metalness > 0.0 {
                    self.albedo
                } else {
                    Vec3::one()
                };
            } else {
                // Transmission (Refraction)
                let refraction_ratio = if is_front_face {
                    1.0 / self.ior
                } else {
                    self.ior
                };

                // Perform refraction calculation only when necessary
                if let Some(refracted_direction) =
                    refract(unit_direction, normal, refraction_ratio)
                {
                    scattered_direction =
                        refracted_direction + roughness * rng.random_in_unit_sphere();
                    attenuation = Vec3::one(); // Assume no attenuation for transmission
                } else {
                    // Total internal reflection
                    let reflected = reflect(unit_direction, normal);
                    scattered_direction = reflected + roughness * rng.random_in_unit_sphere();
                    attenuation = if metalness > 0.0 {
                        self.albedo
                    } else {
                        Vec3::one()
                    };
                }
            }
        }

        (scattered_direction.normalize(), attenuation)
    }

    // Schlick approximation for Fresnel effect
    fn schlick(&self, cosine: f32, ior: f32) -> f32 {
        let r0 = ((1.0 - ior) / (1.0 + ior)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}


// Helper function to reflect a vector
fn reflect(v: Vec3<f32>, n: Vec3<f32>) -> Vec3<f32> {
    v - 2.0 * v.dot(n) * n
}

fn refract(v: Vec3<f32>, n: Vec3<f32>, eta: f32) -> Option<Vec3<f32>> {
    let cos_theta = (-v).dot(n).min(1.0);
    let sin_theta2 = 1.0 - cos_theta * cos_theta;
    let eta2 = eta * eta;
    if eta2 * sin_theta2 > 1.0 {
        None // Total internal reflection
    } else {
        let r_out_perp = eta * (v + cos_theta * n);
        let r_out_parallel = -((1.0 - r_out_perp.length_sq()).abs().sqrt()) * n;
        Some(r_out_perp + r_out_parallel)
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
            emission_strength: 0.,
        }
    }
}