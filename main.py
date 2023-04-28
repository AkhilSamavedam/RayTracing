import numpy as np
import matplotlib.pyplot as plt


class Object:
    def __init__(self, center, ambient, diffuse, specular, shininess=None, reflection=None):
        self.center = np.array(center)
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.shininess = shininess
        self.reflection = reflection


class Sphere(Object):
    # We are using Blinn Phong to render the spheres using all properties listed below

    def __init__(self, center, radius, ambient, diffuse, specular, shininess, reflection):
        super().__init__(center, ambient, diffuse, specular, shininess, reflection)
        self.radius = radius


def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj.center, obj.radius, ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for i, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[i]
    return nearest_object, min_distance


width = 300
height = 200

max_depth = 3

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom



objects = [
    Sphere([-.2, 0, -1], 0.7, [0, 0, 0.1], [0, 0, 0.7], [1, 1, 1], 100, 0.5),
    Sphere([0.1, -0.3, 0],  0.1, [0, 0.1, 0], [0, 0.7, 0], [1, 1, 1], 100, 0.5),
    Sphere([-0.3, 0, 0], 0.15, [0.1, 0, 0.1], [0.7, 0, 0.7], [1, 1, 1], 100, 0.5),
    Sphere([-.5, -.5, -.5], 0.15, [0.1, 0, 0], [0.7, 0, 0], [1, 1, 1], 100, 0.5),
    Sphere([-.2, 0.5, -0.2], 0.2, [.1, .1, 0], [.7, .7, 0], [1, 1, 1], 100, 0.5),
    Sphere([0, -9000, 0], 9000-0.7, [0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [1, 1, 1], 100, 0.5) # Large Screen
]

light = Object([5, 5, 5], [1, 1, 1], [1, 1, 1], [1, 1, 1])


image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        # screen is on origin
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1

        for k in range(max_depth):
            # check for intersections
            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                break

            intersection = origin + min_distance * direction
            normal_to_surface = normalize(intersection - nearest_object.center)
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(light.center - shifted_point)

            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(light.center - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                break

            illumination = np.zeros((3))

            # ambient
            illumination += nearest_object.ambient * light.ambient

            # diffuse
            illumination += nearest_object.diffuse * light.diffuse * np.dot(intersection_to_light, normal_to_surface)

            # specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object.specular * light.specular * np.dot(normal_to_surface, H) ** (nearest_object.shininess / 4)

            # reflection
            color += reflection * illumination
            reflection *= nearest_object.reflection

            # next stage of reflection
            origin = shifted_point
            direction = reflected(direction, normal_to_surface)

        image[i, j] = np.clip(color, 0, 1)
    print("%d/%d" % (i + 1, height))

plt.imsave('image.png', image)