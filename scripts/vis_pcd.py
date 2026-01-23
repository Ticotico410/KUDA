from pathlib import Path
import open3d as o3d

pcd_dir = Path("/home/nikolaraicevic/ycb_ws/KUDA/logs/pcd")
for path in sorted(pcd_dir.glob("*.pcd")):
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.is_empty():
        o3d.visualization.draw_geometries([pcd], window_name=path.name)
