# This code calculates gap distances between two OTS segments (FreeSurfer labels).
# It measures the lengths of each OTS segment and saves all paths as seperate label files, as well as entering data into a CSV file. 
# "ROI" term within this script refers to the current FreeSurfer label, and "Spine" refers to its length.
# Note that the target_labels and sub_id naming conventions need to be adjusted (see below)!

import nibabel as nib
import numpy as np
import os
from scipy.sparse.csgraph import dijkstra, shortest_path
from scipy.sparse import csr_matrix
import csv

hm = "lh" #lh for the left hemisphere and rh for the right hemisphere
target_labels = [f"{hm}.OTS_{x}" for x in range(1, 8)] # given that the number of labels was within the 7 number limit
target_folder_path = "/path/to/the/target/folder" # where subject folders reside


def build_adjacency_matrix(coords, faces):

    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    
    unique_edges = np.unique(edges, axis=0)
    
    v0, v1 = unique_edges[:, 0], unique_edges[:, 1]
    dist = np.linalg.norm(coords[v0] - coords[v1], axis=1)

    # creating an undirected graph (adding both directions for Dijkstra algorithm)
    sources = np.concatenate([v0, v1])
    targets = np.concatenate([v1, v0])
    weights = np.concatenate([dist, dist])

    return csr_matrix((weights, (sources, targets)), shape=(len(coords), len(coords)))

def compute_vertex_areas(coords, faces):
    # calculates surface area assigned to each vertex (1/3 of adjacent face areas)
    v0, v1, v2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    vertex_areas = np.zeros(len(coords))
    np.add.at(vertex_areas, faces[:, 0], face_areas / 3.0)
    np.add.at(vertex_areas, faces[:, 1], face_areas / 3.0)
    np.add.at(vertex_areas, faces[:, 2], face_areas / 3.0)
    return vertex_areas

def save_path_as_label(vertex_indices, coords, output_path):
    # saving the list of vertex indices as a Freesurfer .label file
    with open(output_path, "w") as f:
        f.write("#!ascii label  , from subject  \n")
        f.write(f"{len(vertex_indices)}\n")
        for idx in vertex_indices:
            x, y, z = coords[idx]
            f.write(f"{idx}  {x:.3f}  {y:.3f}  {z:.3f}  0.000000\n")

def compute_geodesic_and_save_spine(coords, faces, roi_indices, label_dir, label_name):
    # finding the longest geodesic path within the OTS segment and later saving it as a spine label
    if len(roi_indices) < 2: return 0.0
    
    global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(roi_indices)}
    local_to_global = {l_idx: g_idx for l_idx, g_idx in enumerate(roi_indices)}
    in_roi = np.zeros(len(coords), dtype=bool)
    in_roi[roi_indices] = True

    # de-duplicating edges within the OTS segment (otherwise all distances will be twice longer, which is not correct)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    
    mask = in_roi[edges[:, 0]] & in_roi[edges[:, 1]]
    roi_edges = edges[mask]
    if len(roi_edges) == 0: return 0.0

    weights = np.linalg.norm(coords[roi_edges[:, 0]] - coords[roi_edges[:, 1]], axis=1)
    local_u = np.array([global_to_local[u] for u in roi_edges[:, 0]])
    local_v = np.array([global_to_local[v] for v in roi_edges[:, 1]])
    
    K = len(roi_indices)
    adj_matrix = csr_matrix((np.concatenate([weights, weights]), 
                            (np.concatenate([local_u, local_v]), np.concatenate([local_v, local_u]))), 
                            shape=(K, K))

    dist_matrix, predecessors = shortest_path(csgraph=adj_matrix, directed=False, return_predecessors=True)
    dist_matrix[np.isinf(dist_matrix)] = -1
    max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    
    path_indices = []
    curr = max_idx[1]
    while curr != -9999:
        path_indices.append(local_to_global[curr])
        if curr == max_idx[0]: break
        curr = predecessors[max_idx[0], curr]

    spine_path = os.path.join(label_dir, f"{label_name}_geodesic_spine.label")
    save_path_as_label(path_indices, coords, spine_path)
    return dist_matrix[max_idx]


list_of_subject_folders = [f for f in os.listdir(target_folder_path) 
                            if os.path.isdir(os.path.join(target_folder_path, f)) and f.startswith("sub-")]

for subject in list_of_subject_folders:
    sub_id = subject.replace("sub-", "").replace("_to_recon_done", "") # has to be adjusted for each project!
    surf_path = f'{target_folder_path}/{subject}/surf/{hm}.white'
    label_dir = f'{target_folder_path}/{subject}/label'
    
    if not os.path.exists(surf_path):
        continue

    print(f"\n--- Processing Subject: {sub_id} ---")
    coords, faces = nib.freesurfer.read_geometry(surf_path)
    
    graph = build_adjacency_matrix(coords, faces)
    v_areas = compute_vertex_areas(coords, faces)

    found_labels = [l for l in target_labels if os.path.exists(os.path.join(label_dir, f"{l}.label"))]
    
    roi_stats = {}
    for l_name in found_labels:
        indices = nib.freesurfer.read_label(os.path.join(label_dir, f"{l_name}.label"))
        spine_len = compute_geodesic_and_save_spine(coords, faces, indices, label_dir, l_name)
        roi_stats[l_name] = {
            'area': np.sum(v_areas[indices]),
            'spine': spine_len,
            'indices': indices
        }

    output_csv = f"{label_dir}/{sub_id}_{hm}_roi_morphology_results.csv"
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Identifier', 'ROI_A', 'ROI_B', 'Gap_Dist_mm', 'Area_A_mm2', 'Area_B_mm2', 'Spine_A_mm', 'Spine_B_mm'])

        if len(found_labels) < 2:
            continue

        for i in range(len(found_labels) - 1):
            n1, n2 = found_labels[i], found_labels[i+1]
            roi1_indices = roi_stats[n1]['indices']
            roi2_indices = roi_stats[n2]['indices']
            
            # Dijkstra from all nodes in ROI 1
            dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, 
                                               indices=roi1_indices, return_predecessors=True)
            
            dists_to_r2 = dist_matrix[:, roi2_indices]
            min_dist = np.min(dists_to_r2)
            
            # mapping back to find the vertices that form the gap
            idx_in_r1, idx_in_r2 = np.unravel_index(np.argmin(dists_to_r2), dists_to_r2.shape)
            v_start = roi1_indices[idx_in_r1]
            v_end = roi2_indices[idx_in_r2]

            # tracing the gap path
            gap_path = []
            curr = v_end
            while curr != -9999:
                gap_path.append(curr)
                if curr == v_start: break
                curr = predecessors[idx_in_r1, curr]
            
            gap_label_path = os.path.join(label_dir, f"{hm}.gap_{n1}_to_{n2}.label")
            save_path_as_label(gap_path, coords, gap_label_path)

            writer.writerow([
                sub_id, n1, n2, 
                f"{min_dist:.4f}", 
                f"{roi_stats[n1]['area']:.4f}", f"{roi_stats[n2]['area']:.4f}",
                f"{roi_stats[n1]['spine']:.4f}", f"{roi_stats[n2]['spine']:.4f}"
            ])

print("\nProcessing Complete.")
