use crate::random::Random;
use crate::ray::{Interval, Ray};
use crate::vec3;
use crate::vectors::{Transform, Vec3};


#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone)]
pub enum ObjectType{
    Sphere,
}

#[derive(Debug, Clone)]
pub struct Object{
    transform: Transform,
    moved_transform: Option<Transform>,
    object_type: ObjectType,
    radius: f32,
    pub material: Material,
}
impl Object{
    pub fn new(transform: Transform, object_type: ObjectType, radius: f32) -> Object {
        Object{transform, object_type, radius, material: Material::default(), moved_transform: None}
    }
    pub fn new_sphere(transform: Transform, radius: f32) -> Object {
        Object{transform, object_type: ObjectType::Sphere, radius, material: Material::default(), moved_transform: None}
    }
    pub fn new_sphere_with_material(transform: Transform, radius: f32, material: Material) -> Object {
        Object{transform, object_type: ObjectType::Sphere, radius, material, moved_transform: None}
    }
    pub fn with_movement(mut self, move_to: Transform) -> Self{
        self.moved_transform = Some(move_to);
        self
    }

    fn collide_sphere(&self, ray: &Ray) -> HitRecord {
        let mut hit = HitRecord::default();
        let mut obj_transform = self.transform;

        if let Some(moved) = self.moved_transform{
            obj_transform = obj_transform.lerp(moved, ray.time)
        }

        // Initialize the hit record
        let oc = obj_transform.position - ray.origin;
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

        let outward_normal = (hit.position - obj_transform.position) / self.radius;
        hit.set_face_normal(&ray, outward_normal);

        hit
    }
    pub(crate) fn collide(&self, ray: &Ray) -> HitRecord{
        match self.object_type {
            ObjectType::Sphere => {self.collide_sphere(ray)},
        }
    }
    pub fn compute_bounding_volume(&self) -> BoundingVolume{
        match &self.object_type {
            ObjectType::Sphere => {
                let min = self.transform.position - vec3!(self.radius);
                let max = self.transform.position + vec3!(self.radius);
                let bv = BoundingVolume::from_to(min, max);

                if let Some(moved) = self.moved_transform{
                    let min = moved.position - vec3!(self.radius);
                    let max = moved.position + vec3!(self.radius);
                    let bv2 = BoundingVolume::from_to(min, max);
                    return BoundingVolume::new_enclosing(&[bv,bv2])
                }
                bv
            }
        }
    }
}



impl Default for Object{
    fn default() -> Self {
        Self{
            transform: Transform::at(Vec3::zero()),
            object_type: ObjectType::Sphere,
            radius: 1.0,
            material: Material::default(),
            moved_transform: None,
        }
    }
}



#[derive(Copy, Clone)]
struct BoundingVolume{
    x: Interval,
    y: Interval,
    z: Interval,
}
impl BoundingVolume {
    pub fn new(x: Interval, y: Interval, z: Interval) -> BoundingVolume {
        BoundingVolume{x, y, z}
    }
    pub fn from_to(min: Vec3<f32>, max: Vec3<f32>) -> BoundingVolume {
        let x = if max.x >= min.x {Interval::new(min.x, max.x)} else {Interval::new(max.x, min.x)};
        let y = if max.y >= min.y {Interval::new(min.y, max.y)} else {Interval::new(max.y, min.y)};
        let z = if max.z >= min.z {Interval::new(min.z, max.z)} else {Interval::new(max.z, min.z)};
        BoundingVolume{x, y, z}
    }

    pub fn new_enclosing(volumes: &[BoundingVolume]) -> BoundingVolume {
        if volumes.is_empty() {
            panic!("Cannot create an enclosing bounding volume from an empty list.");
        }

        // Initialize min and max bounds with the first bounding volume's values.
        let mut min_x = volumes[0].x.min;
        let mut max_x = volumes[0].x.max;
        let mut min_y = volumes[0].y.min;
        let mut max_y = volumes[0].y.max;
        let mut min_z = volumes[0].z.min;
        let mut max_z = volumes[0].z.max;

        // Expand the bounds to include all volumes.
        for volume in volumes.iter().skip(1) {
            min_x = min_x.min(volume.x.min);
            max_x = max_x.max(volume.x.max);
            min_y = min_y.min(volume.y.min);
            max_y = max_y.max(volume.y.max);
            min_z = min_z.min(volume.z.min);
            max_z = max_z.max(volume.z.max);
        }

        BoundingVolume {
            x: Interval::new(min_x, max_x),
            y: Interval::new(min_y, max_y),
            z: Interval::new(min_z, max_z),
        }
    }

    pub fn split_volumes(
        volumes: &[(usize, BoundingVolume)],
        enclosing_volume: BoundingVolume,
    ) -> (Vec<(usize, BoundingVolume)>, Vec<(usize, BoundingVolume)>) {
        if volumes.is_empty() {
            panic!("Cannot split an empty list of volumes.");
        }

        // Step 1: Determine the longest axis
        let x_length = enclosing_volume.x.max - enclosing_volume.x.min;
        let y_length = enclosing_volume.y.max - enclosing_volume.y.min;
        let z_length = enclosing_volume.z.max - enclosing_volume.z.min;

        let longest_axis = if x_length >= y_length && x_length >= z_length {
            0 // X-axis
        } else if y_length >= z_length {
            1 // Y-axis
        } else {
            2 // Z-axis
        };

        // Step 2: Sort the volumes based on their center along the longest axis
        let mut sorted_volumes = volumes.to_vec();
        sorted_volumes.sort_by(|a, b| {
            let a_center = match longest_axis {
                0 => (a.1.x.min + a.1.x.max) / 2.0, // Use the x-axis center
                1 => (a.1.y.min + a.1.y.max) / 2.0, // Use the y-axis center
                _ => (a.1.z.min + a.1.z.max) / 2.0, // Use the z-axis center
            };
            let b_center = match longest_axis {
                0 => (b.1.x.min + b.1.x.max) / 2.0,
                1 => (b.1.y.min + b.1.y.max) / 2.0,
                _ => (b.1.z.min + b.1.z.max) / 2.0,
            };
            a_center.partial_cmp(&b_center).unwrap()
        });

        // Step 3: Split the volumes into two roughly equal groups
        let mid = sorted_volumes.len() / 2;
        let group1 = sorted_volumes[..mid].to_vec();
        let group2 = sorted_volumes[mid..].to_vec();

        (group1, group2)
    }
    
    fn interval_axis(&self, n: usize) -> Interval {
        match n {
            1 => {self.y}
            2 => {self.z}
            _ => {self.x}
        }
    }
    pub fn hit(&self, mut ray: Ray) -> bool{
        for axis in 0..3{
            assert!(axis < 3, "Axis index out of bounds");
            let a_itv = self.interval_axis(axis);
            let direction = match axis {
                1 => ray.direction.y,
                2 => ray.direction.z,
                _ => ray.direction.x
            };
            let a_d_inv = match direction {
                0.0 => return self.x.contains(ray.origin.x), // For example, check if ray origin is inside bounds
                _ => 1. / direction,
            };

            let r_o_ax = match axis {
                1 => ray.origin.y,
                2 => ray.origin.z,
                _ => ray.origin.x
            };
            let t0 = (a_itv.min - r_o_ax) * a_d_inv;
            let t1 = (a_itv.max - r_o_ax) * a_d_inv;
            if t0 < t1{
                if t0 > ray.interval.min {ray.interval.min = t0}
                if t1 < ray.interval.max {ray.interval.max = t1}
            }
            else {
                if t1 > ray.interval.min {ray.interval.min = t1}
                if t0 < ray.interval.max {ray.interval.max = t0}
            }
            if ray.interval.max <= ray.interval.min{
                return false;
            }
        }
        true
    }
}
struct BVTreeNode{
    bounding_volume: BoundingVolume,
    children: Option<(Box<BVTreeNode>, Box<BVTreeNode>)>,
    object_index: Option<usize>
}


struct BVTree{
    root: Option<BVTreeNode>,
}

impl BVTree {
    pub fn build(objects: &[Object]) -> BVTree {
        let bounding_volumes = objects
            .iter()
            .enumerate()
            .map(|(index, object)| (index, object.compute_bounding_volume()))
            .collect::<Vec<_>>();
        let root = BVTree::build_tree(&bounding_volumes);
        BVTree{root: Some(root)}
    }
    fn build_tree(bounding_volumes: &[(usize, BoundingVolume)]) -> BVTreeNode{
        if bounding_volumes.len() == 1 {
            BVTreeNode{
                bounding_volume: bounding_volumes[0].1.clone(),
                children: None,
                object_index: Some(bounding_volumes[0].0)
            }
        }
        else {
            let just_volumes = &*bounding_volumes.iter().map(|x| {x.1.clone()}).collect::<Vec<BoundingVolume>>();
            let enclosing = BoundingVolume::new_enclosing(just_volumes);
            let (group1, group2) = BoundingVolume::split_volumes(bounding_volumes, enclosing);
            let left_node = BVTree::build_tree(&group1);
            let right_node = BVTree::build_tree(&group2);
            BVTreeNode{
                bounding_volume: enclosing,
                children: Some((Box::new(left_node), Box::new(right_node))),
                object_index: None
            }

        }

    }
}

pub struct World{
    pub(crate) objects: Vec<Object>,
    bvtree: Option<BVTree>,
}
impl World {
    pub fn new() -> World {
        World { objects: vec![], bvtree: None }
    }
    pub fn add(&mut self, object: Object) {
        self.objects.push(object);
    }

    pub fn build_bvtree(&mut self){
        self.bvtree = Some(BVTree::build(&*self.objects))
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
        let moving_ball = Object::new_sphere_with_material(
            Transform::at(vec3!(5., 2., 5.)),
            0.5,
            Material::new_diffuse(vec3!(0.2, 0.8, 0.8)),
        ).with_movement(Transform::at(vec3!(5.5, 1.7, 5.)));
        //println!("{:?}", moving_ball);

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
        self.add(moving_ball)
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

    pub fn movement_test_scene(&mut self){
        let ground = Object::new_sphere_with_material(
            Transform::at(vec3!(0., -11., 5.)),
            10.,
            Material::new_diffuse(Vec3::new(1., 1., 1.)),
        );
        let moving_ball = Object::new_sphere_with_material(
            Transform::at(vec3!(-1., 0., 5.)),
            1.,
            Material::new_diffuse(Vec3::new(1., 1., 1.)),
        ).with_movement(
            Transform::at(vec3!(-0.5, 0., 5.)),
        );
        self.add(moving_ball);
        self.add(ground);
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

    /*pub fn collide(&self, ray: &Ray) -> HitRecord{
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
    }*/
    /// Find the closest collision in the scene using the BVTree.
    pub fn collide(&self, ray: &mut Ray) -> HitRecord {
        let mut closest_hit = HitRecord::default();
        closest_hit.t = ray.interval.max; // Start with the maximum interval

        // If no BVT is available, fall back to brute-force
        if let Some(ref bvtree) = self.bvtree {
            // Start traversal from the root node
            if let Some(ref root) = bvtree.root {
                Self::traverse_bvtree(root, ray, &mut closest_hit, &self.objects);
            }
        } else {
            // Brute-force approach
            for object in &self.objects {
                let hit = object.collide(ray);
                if hit.hit && hit.t < closest_hit.t {
                    closest_hit = hit;
                }
            }
        }

        closest_hit
    }

    /// Recursive traversal of the BVTree
    fn traverse_bvtree(
        node: &BVTreeNode,
        ray: &mut Ray,
        closest_hit: &mut HitRecord,
        objects: &[Object],
    ) {
        // Check if the ray intersects the node's bounding volume
        if !node.bounding_volume.hit(*ray){
            return; // Skip this node entirely
        }

        match &node.children {
            Some((left_child, right_child)) => {
                // Inner node: Traverse both children
                Self::traverse_bvtree(left_child, ray, closest_hit, objects);
                Self::traverse_bvtree(right_child, ray, closest_hit, objects);
            }
            None => {
                // Leaf node: Test objects for collision
                if let Some(object_index) = node.object_index {
                    let hit = objects[object_index].collide(ray);
                    if hit.hit && hit.t <= closest_hit.t {
                        *closest_hit = hit;
                    }
                }
            }
        }
    }
}


#[derive(Debug, Copy, Clone)]
pub struct Material{
    pub albedo: Vec3<f32>,
    metalness: f32,
    roughness: f32,
    ior: f32,
    pub emission: Vec3<f32>,
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
            let _specular_prob = specular_prob / sum;

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
            let _transmission_prob = transmission_prob / sum;

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
        let r_out_parallel = -(1.0 - r_out_perp.length_sq()).abs().sqrt() * n;
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