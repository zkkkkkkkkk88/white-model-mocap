"""Bind the approved rigid limb blocks at their actual shared mesh pivots."""

from itertools import product

from mathutils import Matrix, Vector


def mean(points):
    return sum(points, Vector()) / len(points)


def face_planes(mesh, points):
    center = mean(points)
    for face in mesh.data.polygons:
        a, b, c = (points[i] for i in face.vertices[:3])
        normal = (b - a).cross(c - a).normalized()
        if normal.dot(center - a) > 0:
            normal.negate()
        yield a, normal


def seam_pivot(meshes, geometry_rows, upper_axis, lower_axis, inside_candidate):
    cap_centers = []
    for axis, geo, is_upper in zip((upper_axis, lower_axis), geometry_rows, (True, False), strict=True):
        points = sorted(geo["vertices"], key=axis.dot, reverse=is_upper)
        cap_centers.append(mean(points[:4]))
    middle = mean(cap_centers)
    low = [max(geo["bounds"][0][axis] for geo in geometry_rows) for axis in range(3)]
    high = [min(geo["bounds"][1][axis] for geo in geometry_rows) for axis in range(3)]
    planes = [plane for mesh, geo in zip(meshes, geometry_rows, strict=True)
              for plane in face_planes(mesh, geo["vertices"])]
    # Prefer the middle of the measured caps, with positive interior clearance on both sides.
    # A fixed tie tolerance prevents minute mirror-side floating point differences becoming offsets.
    step = [(end - start) / 10 for start, end in zip(low, high, strict=True)]
    minimum_clearance = min(step) / 5
    if minimum_clearance <= 0:
        raise ValueError("白模肢体的初始接缝没有共同体积")
    ranked = []
    for offsets in product(range(1, 10), repeat=3):
        point = Vector([start + i * width for start, i, width in zip(low, offsets, step, strict=True)])
        clearance = min(normal.dot(origin - point) for origin, normal in planes)
        if clearance >= minimum_clearance:
            rank = round((point - middle).length_squared, 7), -round(clearance, 6)
            ranked.append((rank, point))
    for _, point in sorted(ranked, key=lambda row: row[0]):
        if all(inside_candidate(point, geo) is True for geo in geometry_rows):
            return point
    raise ValueError("无法在白模原有接缝内确定共同关节轴心")


def bind_block(context, rig, mesh, bone, pivot, world_vertices):
    rotation = (rig.matrix_world @ bone.matrix).to_quaternion().normalized()
    neutral = Matrix.LocRotScale(pivot, rotation, Vector((1, 1, 1)))
    if mesh.data.users > 1:
        mesh.data = mesh.data.copy()
    inverse = neutral.inverted()
    # Freeze the visible, evaluated library shape before disabling its deformer.
    for vertex, world in zip(mesh.data.vertices, world_vertices, strict=True):
        vertex.co = inverse @ world
    mesh.data.update()
    for modifier in mesh.modifiers:
        if modifier.type == "ARMATURE" and modifier.object == rig:
            modifier.show_viewport = False
            modifier.show_render = False
    mesh.parent = rig
    mesh.parent_type = "BONE"
    mesh.parent_bone = bone.name
    mesh.matrix_parent_inverse = Matrix.Identity(4)
    context.view_layer.update()
    mesh.matrix_world = neutral
    context.view_layer.update()


def register_joint_meshes(context, rig, parts, parents, controls):
    """Use existing block overlap as the hinge, keeping the rest silhouette and FK editability."""
    from collision_scan import geometry, inside_candidate

    if rig.data.pose_position != "POSE":
        raise ValueError("白模关节绑定需要使用骨架库默认求值姿态")
    context.view_layer.update()
    limb_parts = set(parents) | set(parents.values())
    neutral = {part: geometry(parts[part]) for part in limb_parts}
    max_error = 0.0
    for lower, upper in parents.items():
        meshes = parts[upper], parts[lower]
        bones = [rig.pose.bones[controls[part]] for part in (upper, lower)]
        geometries = neutral[upper], neutral[lower]
        if any(len(geo["vertices"]) != 8 or not geo["closed"] for geo in geometries):
            raise ValueError(f"关节刚性绑定要求已验证的白模块体：{upper}、{lower}")
        axes = [(rig.matrix_world.to_3x3() @ (bone.tail - bone.head)).normalized() for bone in bones]
        pivot = seam_pivot(meshes, geometries, *axes, inside_candidate)
        for mesh, bone, geo in zip(meshes, bones, geometries, strict=True):
            bind_block(context, rig, mesh, bone, pivot, geo["vertices"])
        constraint = meshes[1].constraints.new("COPY_LOCATION")
        constraint.name = "白模关节共用轴心"
        constraint.target = meshes[0]
        constraint.target_space = constraint.owner_space = "WORLD"
        context.view_layer.update()
        for mesh, old in zip(meshes, geometries, strict=True):
            current = geometry(mesh)["vertices"]
            max_error = max(max_error, max((a - b).length for a, b in zip(old["vertices"], current, strict=True)))
    if max_error > 1e-5:
        raise ValueError(f"关节绑定改变了白模静止形状：{max_error:.5g}")
    return {"joints": len(parents), "max_rest_vertex_error": max_error,
            "reference_pose": "library_default_evaluated_pose"}
