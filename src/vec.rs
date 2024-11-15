pub struct Vec3{
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl Vec3{
    pub fn new(x: f32, y: f32, z: f32) -> Vec3{
        Vec3{x,y,z}
    }
    pub fn splat(x: f32) -> Vec3{
        Self::new(x,x,x)
    }
    pub fn length(&self) -> f32{
        (self.x*self.x + self.y*self.y + self.z*self.z).sqrt()
    }
    pub fn dot(&self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

}
impl std::fmt::Display for Vec3{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error>{
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}
impl std::ops::Add for Vec3{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}
impl std::ops::AddAssign for Vec3{
    fn add_assign(&mut self, other: Self){
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}
impl std::ops::Sub for Vec3{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}
impl std::ops::SubAssign for Vec3{
    fn sub_assign(&mut self, other: Self){
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}
impl std::ops::Mul<f32> for Vec3{
    type Output = Self;
    fn mul(self, n :f32) -> Self {
        Self::new(self.x * n,self.y * n,self.z * n)
    }
}
impl std::ops::Mul for Vec3 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}
impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, n : Vec3) -> Vec3 {
        Vec3::new(self * n.x,self * n.y,self * n.z)
    }
}

impl std::ops::MulAssign for Vec3{
    fn mul_assign(&mut self, n: f32){
        self.x *= n;
        self.y *= n;
        self.z *= n;
    }
}
impl std::ops::Div<f32> for Vec3{
    type Output = Self;
    fn div(self, n :f32) -> Self {
        Self::new(self.x / n,self.y / n,self.z / n)
    }
}

impl std::ops::DivAssign for Vec3{
    fn div_assign(&mut self, n:f32){
        self.x /= n;
        self.y /= n;
        self.z /= n;
    }
}
impl std::ops::Neg for Vec3{
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}


macro_rules! vec3 {
    ($x:expr, $y:expr, $z:expr) => {
        Vec3::new($x, $y, $z)
    };
    ($x:expr) => {
        Vec3::splat($x)
    };
}