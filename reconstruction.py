# TODO limpiar imports innecesarios

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree, distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize
import nibabel as nib
import json


class VascularTreeReconstruction:
    """
    Reconstruct vascular tree from skeleton points using CCO principles
    """

    def __init__(self, skeleton_points, gamma=3.0, mu=3.6e-3,
                 Q_perf=0.125, P_out=60, P_in=100):
        """
        Parameters:
        -----------
        skeleton_points : np.array, shape (N, 3)
            3D coordinates of skeleton points
        gamma : float
            Murray's law exponent (default: 3.0)
        mu : float
            Blood viscosity in Pa·s (default: 3.6e-3 Pa·s = 3.6 cP)
        Q_perf : float
            Flow at each terminal in mL/min (default: 0.125)
        P_out : float
            Pressure at terminals in mmHg (default: 60)
        P_in : float
            Pressure at root in mmHg (default: 100)
        """
        self.points = np.array(skeleton_points)
        self.gamma = gamma
        self.mu = mu
        self.Q_perf = Q_perf / 60000  # Convert mL/min to mm³/s
        self.P_out = P_out * 133.322  # Convert mmHg to Pa
        self.P_in = P_in * 133.322
        self.kappa = 8 * mu / np.pi
        self.xi = self.Q_perf / (self.P_in - self.P_out)

        self.graph = None
        self.root = None
        self.tree_structure = None

    def build_graph_from_skeleton(self, k_neighbors=10, max_edge_length=None):
        """
        Step 1-2: Build connected graph from skeleton points
        Uses k-nearest neighbors + MST to ensure connectivity

        Parameters:
        -----------
        k_neighbors : int
            Number of nearest neighbors to consider
        max_edge_length : float
            Maximum allowed edge length (for pruning)
        """
        print("Building graph from skeleton points...")
        n_points = len(self.points)

        # Build k-NN graph
        tree = cKDTree(self.points)

        # Create adjacency matrix
        distances = np.zeros((n_points, n_points))
        distances[:] = np.inf

        for i in range(n_points):
            dists, indices = tree.query(
                self.points[i], k=min(k_neighbors+1, n_points))
            for j, idx in enumerate(indices[1:]):  # Skip self
                distances[i, idx] = dists[j+1]
                distances[idx, i] = dists[j+1]  # Symmetric

        # Build MST to ensure single connected component and remove cycles
        mst = minimum_spanning_tree(distances)

        # Convert to networkx graph
        self.graph = nx.Graph()

        # Add nodes with positions
        for i, pos in enumerate(self.points):
            self.graph.add_node(i, pos=pos)

        # Add edges from MST
        mst_coo = mst.tocoo()
        for i, j, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
            self.graph.add_edge(i, j, length=weight)

        print(
            f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        # Prune short branches
        if max_edge_length:
            self._prune_branches(max_edge_length)

        return self.graph

    def _prune_branches(self, min_length):
        """Remove short terminal branches"""
        pruned = True
        while pruned:
            pruned = False
            endpoints = [n for n in self.graph.nodes()
                         if self.graph.degree(n) == 1]

            for node in endpoints:
                neighbor = list(self.graph.neighbors(node))[0]
                edge_length = self.graph[node][neighbor]['length']

                if edge_length < min_length:
                    self.graph.remove_node(node)
                    pruned = True

        print(
            f"After pruning: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def find_root_candidates(self, n_candidates=5, method="highest_z"):
        """
        Step 4: Identify potential root points

        Returns candidates based on:
        - Boundary points (degree 1)
        - Spatial location (e.g., superior, medial)
        - High centrality
        """
        candidates = []

        # Get all endpoints (degree 1)
        endpoints = [n for n in self.graph.nodes()
                     if self.graph.degree(n) == 1]

        if len(endpoints) == 0:
            print("Warning: No endpoints found, using highest degree nodes")
            degrees = dict(self.graph.degree())
            endpoints = sorted(degrees, key=degrees.get, reverse=True)[:10]

        endpoint_positions = np.array([self.points[i] for i in endpoints])

        if method == "highest_z":
            # Strategy 1: Highest z-coordinate (superior position for hepatic vein)
            z_coords = endpoint_positions[:, 2]
            top_z_indices = np.argsort(z_coords)[-n_candidates:]
            candidates.extend([endpoints[i] for i in top_z_indices])

            print(
                f"Found {len(candidates)} candidates from superior endpoints")

        if method == "lowest_z":
            # Strategy 1: Lowest z-coordinate (inferior position for hepatic vein)
            z_coords = endpoint_positions[:, 2]
            bottom_z_indices = np.argsort(z_coords)[:n_candidates]
            candidates.extend([endpoints[i] for i in bottom_z_indices])

            print(
                f"Found {len(candidates)} candidates from inferior endpoints")

        if method == "centrality":
            # Strategy 2: Most central points (low eccentricity)
            if len(self.graph) > 10:
                try:
                    # Get largest connected component
                    if not nx.is_connected(self.graph):
                        largest_cc = max(
                            nx.connected_components(self.graph), key=len)
                        subgraph = self.graph.subgraph(largest_cc)
                    else:
                        subgraph = self.graph

                    centrality = nx.closeness_centrality(subgraph)
                    central_nodes = sorted(centrality, key=centrality.get, reverse=True)[
                        :n_candidates]
                    candidates.extend(
                        [n for n in central_nodes if n in endpoints])
                except:
                    pass

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        print(f"Found {len(unique_candidates)} root candidates")
        return unique_candidates[:n_candidates]

    def orient_tree_from_root(self, root):
        """
        Step 5a: Create directed tree structure from root
        Returns tree with parent-child relationships
        """
        if not nx.is_connected(self.graph):
            # Get largest connected component containing root
            for component in nx.connected_components(self.graph):
                if root in component:
                    subgraph = self.graph.subgraph(component).copy()
                    break
        else:
            subgraph = self.graph

        # BFS from root to create directed tree
        tree = nx.DiGraph()
        visited = {root}
        queue = [root]

        # Copy node attributes
        for node in subgraph.nodes():
            tree.add_node(node, pos=self.points[node])

        while queue:
            current = queue.pop(0)

            for neighbor in subgraph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # Add directed edge from parent to child
                    edge_data = subgraph[current][neighbor]
                    tree.add_edge(current, neighbor, **edge_data)
                    queue.append(neighbor)

        return tree

    def compute_tree_parameters(self, tree, root):
        """
        Step 5b: Compute L_i (number of terminals in subtree)
        """
        # Find all terminal nodes (leaves)
        terminals = [n for n in tree.nodes() if tree.out_degree(n)
                     == 0 and n != root]

        # Compute L_i for each node (number of terminals in subtree)
        L = {}

        def count_terminals(node):
            if tree.out_degree(node) == 0:  # Terminal
                L[node] = 1
                return 1

            total = 0
            for child in tree.successors(node):
                total += count_terminals(child)
            L[node] = total
            return total

        count_terminals(root)

        return L, terminals

    def compute_radii(self, tree, root, L):
        """
        Step 5c: Apply CCO radius formulas (Equations 1-7)
        """
        # Initialize radius dict
        R = {}  # Hydraulic resistance
        beta = {}  # Radius ratio to parent
        rho = {}  # Relative radius to root segment
        radii = {}  # Actual radii

        # Compute resistance R and beta bottom-up
        def compute_resistance(node):
            children = list(tree.successors(node))

            if len(children) == 0:  # Terminal node
                # For terminal: R = kappa * length / r^4
                # We'll initialize with a default and iterate
                R[node] = 1.0  # Will be updated
                return R[node]

            # Get edge length to parent
            parent = list(tree.predecessors(node))
            if parent:
                length = tree[parent[0]][node]['length']
            else:  # Root
                # Estimate length from first child
                if children:
                    length = tree[node][children[0]]['length']
                else:
                    length = 1.0

            # Compute children resistances first (bottom-up)
            child_resistances = []
            for child in children:
                child_resistances.append(compute_resistance(child))

            # Compute beta for each child (Equations 3-4)
            if len(children) == 2:
                child1, child2 = children

                # Alpha ratio (Eq. 4)
                alpha = ((L[child1] / L[node]) *
                         (R[child1] / R[child2])) ** 0.25

                # Beta for each child (Eq. 3)
                beta[child1] = (1 + alpha ** self.gamma) ** (-1/self.gamma)
                beta[child2] = (1 + (1/alpha) ** self.gamma) ** (-1/self.gamma)
            else:
                # Single child or more than 2 children
                for child in children:
                    beta[child] = 1.0

            # Compute resistance (Eq. 6)
            sum_term = sum(beta[child]**4 / R[child] for child in children)
            if sum_term > 0:
                R[node] = self.kappa * length + (1 / sum_term)
            else:
                R[node] = self.kappa * length

            return R[node]

        # Start computation from root
        compute_resistance(root)

        # Compute absolute radii top-down (Eq. 7)
        def compute_radii_recursive(node, parent_radius=None):
            if parent_radius is None:
                # Root radius (Eq. 7)
                n_terminals = L[root]
                r1 = (self.xi * R[root] * n_terminals) ** 0.25
                radii[node] = r1
                rho[node] = 1.0
            else:
                # Child radius
                rho[node] = rho[list(tree.predecessors(node))[0]] * beta[node]
                radii[node] = parent_radius * beta[node]

            # Recurse to children
            for child in tree.successors(node):
                compute_radii_recursive(child, radii[node])

        compute_radii_recursive(root)

        return radii, R, beta, rho

    def optimize_bifurcation(self, tree, node, radii):
        """
        Step 5d: Kamiya optimization for bifurcation point

        Optimizes the position of a bifurcation node to minimize total volume
        """
        # Check if this is a bifurcation
        children = list(tree.successors(node))
        parents = list(tree.predecessors(node))

        if len(children) != 2 or len(parents) != 1:
            return  # Not a bifurcation or is root

        parent = parents[0]
        child1, child2 = children

        # Get positions
        p_parent = self.points[parent]
        p_child1 = self.points[child1]
        p_child2 = self.points[child2]
        p_current = self.points[node]

        # Get radii
        r0 = radii[node]  # Parent segment
        r1 = radii[child1]
        r2 = radii[child2]

        # Flow ratios (proportional to L_i)
        f0 = 1.0  # Normalized
        f1 = r1**3  # From Eq. 10: fi ∝ ri³
        f2 = r2**3

        def objective(x):
            """
            Optimize bifurcation position using Kamiya method
            Based on Equations 9-13 from the paper
            """
            # x is the new position of the bifurcation node
            if len(x) == 2:
                p_bif = np.array([x[0], x[1], p_current[2]])  # 2D
            else:
                p_bif = x

            # Compute lengths
            l0 = np.linalg.norm(p_bif - p_parent)
            l1 = np.linalg.norm(p_child1 - p_bif)
            l2 = np.linalg.norm(p_child2 - p_bif)

            if l0 < 1e-6 or l1 < 1e-6 or l2 < 1e-6:
                return np.array([1e10, 1e10])

            # Pressure drop equations (Eq. 9)
            delta1 = f0 * l0 / r0**4 + f1 * l1 / r1**4
            delta2 = f0 * l0 / r0**4 + f2 * l2 / r2**4

            # Murray's law constraint (Eq. 11)
            r0_expected = (f0 * (r1**6/f1 + r2**6/f2))**(1/3)

            # Residuals (Eq. 12)
            residual1 = delta1 * r1**4 - f0 * l0 * r1**4 / r0**4 - f1 * l1
            residual2 = delta2 * r2**4 - f0 * l0 * r2**4 / r0**4 - f2 * l2

            return np.array([residual1, residual2])

        # Initial guess: current position
        x0 = p_current[:2] if len(p_current) == 3 else p_current

        # Bounds: stay within triangle formed by parent and children
        try:
            result = least_squares(objective, x0, method='lm', max_nfev=100)

            if result.success:
                # Update position
                if len(p_current) == 3:
                    new_pos = np.array(
                        [result.x[0], result.x[1], p_current[2]])
                else:
                    new_pos = result.x

                # Check if new position is reasonable (within triangle + margin)
                max_dist = max(np.linalg.norm(p_parent - p_current),
                               np.linalg.norm(p_child1 - p_current),
                               np.linalg.norm(p_child2 - p_current))

                if np.linalg.norm(new_pos - p_current) < 2 * max_dist:
                    self.points[node] = new_pos
                    tree.nodes[node]['pos'] = new_pos

                    # Update edge lengths
                    tree[parent][node]['length'] = np.linalg.norm(
                        new_pos - p_parent)
                    tree[node][child1]['length'] = np.linalg.norm(
                        p_child1 - new_pos)
                    tree[node][child2]['length'] = np.linalg.norm(
                        p_child2 - new_pos)
        except:
            pass  # Keep original position if optimization fails

    def compute_quality_metrics(self, tree, root, radii, L):
        """
        Step 6: Compute quality metrics for tree evaluation
        """
        metrics = {}

        # 1. Total volume
        total_volume = 0
        for u, v in tree.edges():
            length = tree[u][v]['length']
            radius = radii[v]  # Child radius
            volume = np.pi * radius**2 * length
            total_volume += volume

        metrics['total_volume'] = total_volume

        # 2. Check monotonic radius decrease
        radius_violations = 0
        for node in tree.nodes():
            if node == root:
                continue
            parent = list(tree.predecessors(node))[0]
            if radii[node] > radii[parent]:
                radius_violations += 1

        metrics['radius_violations'] = radius_violations

        # 3. Murray's law compliance at bifurcations
        murray_errors = []
        for node in tree.nodes():
            children = list(tree.successors(node))
            if len(children) == 2:
                r_parent = radii[node]
                r_child1 = radii[children[0]]
                r_child2 = radii[children[1]]

                # Murray's law: r0^γ = r1^γ + r2^γ
                expected = (r_child1**self.gamma + r_child2 **
                            self.gamma)**(1/self.gamma)
                error = abs(r_parent - expected) / r_parent
                murray_errors.append(error)

        metrics['murray_error_mean'] = np.mean(
            murray_errors) if murray_errors else 0
        metrics['murray_error_max'] = np.max(
            murray_errors) if murray_errors else 0

        # 4. Geometric quality - bifurcation angles
        angles = []
        for node in tree.nodes():
            children = list(tree.successors(node))
            if len(children) == 2:
                pos = self.points[node]
                pos1 = self.points[children[0]]
                pos2 = self.points[children[1]]

                v1 = pos1 - pos
                v2 = pos2 - pos

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)
                                              * np.linalg.norm(v2) + 1e-10)
                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                angles.append(angle)

        metrics['mean_bifurcation_angle'] = np.mean(angles) if angles else 0
        metrics['min_bifurcation_angle'] = np.min(angles) if angles else 0

        # 5. Root quality - prefer anatomically superior position
        root_z = self.points[root][2] if len(self.points[root]) == 3 else 0
        max_z = np.max(self.points[:, 2]) if len(self.points[0]) == 3 else 1
        metrics['root_z_score'] = root_z / (max_z + 1e-10)

        return metrics

    def reconstruct(self, k_neighbors=10, n_candidates=5, optimize_bifurcations=True, method="highest_z"):
        """
        Main reconstruction pipeline

        Returns:
        --------
        best_tree : networkx.DiGraph
            The best reconstructed tree
        best_root : int
            Index of the best root node
        best_radii : dict
            Radius for each segment
        metrics : dict
            Quality metrics
        """
        # Step 1-2: Build graph
        if self.graph is None:
            self.build_graph_from_skeleton(k_neighbors=k_neighbors)

        # Step 3: Terminal points (already identified in graph structure)

        # Step 4: Find root candidates
        candidates = self.find_root_candidates(
            n_candidates=n_candidates, method=method)

        print(f"\nEvaluating {len(candidates)} root candidates...")

        best_score = float('inf')
        best_tree = None
        best_root = None
        best_radii = None
        best_metrics = None

        for i, root in enumerate(candidates):
            print(f"\nCandidate {i+1}/{len(candidates)}: node {root}")

            try:
                # Step 5a: Orient tree from this root
                tree = self.orient_tree_from_root(root)

                # Step 5b: Compute L_i
                L, terminals = self.compute_tree_parameters(tree, root)
                print(
                    f"  Terminals: {len(terminals)}, Total nodes: {tree.number_of_nodes()}")

                # Step 5c: Compute radii
                radii, R, beta, rho = self.compute_radii(tree, root, L)

                # Step 5d: Optimize bifurcations (optional)
                if optimize_bifurcations:
                    bifurcations = [n for n in tree.nodes()
                                    if tree.out_degree(n) == 2 and tree.in_degree(n) == 1]
                    print(f"  Optimizing {len(bifurcations)} bifurcations...")
                    # Limit for speed
                    for bif in bifurcations[:min(10, len(bifurcations))]:
                        self.optimize_bifurcation(tree, bif, radii)

                    # Recompute radii after optimization
                    radii, R, beta, rho = self.compute_radii(tree, root, L)

                # Step 6: Evaluate quality
                metrics = self.compute_quality_metrics(tree, root, radii, L)

                print(f"  Metrics: volume={metrics['total_volume']:.2f}, "
                      f"murray_error={metrics['murray_error_mean']:.4f}, "
                      f"radius_violations={metrics['radius_violations']}")

                # Scoring: lower is better
                score = (metrics['total_volume'] / 1000 +  # Normalize volume
                         metrics['murray_error_mean'] * 100 +
                         metrics['radius_violations'] * 10 -
                         metrics['root_z_score'] * 50)  # Prefer higher roots

                if score < best_score:
                    best_score = score
                    best_tree = tree
                    best_root = root
                    best_radii = radii
                    best_metrics = metrics

            except Exception as e:
                print(f"  Failed: {e}")
                continue

        if best_tree is None:
            raise RuntimeError("No valid tree found from any root candidate")

        print(f"\n=== Best tree (root={best_root}) ===")
        for k, v in best_metrics.items():
            print(f"  {k}: {v}")

        self.tree_structure = {
            'tree': best_tree,
            'root': best_root,
            'radii': best_radii,
            'metrics': best_metrics
        }

        return best_tree, best_root, best_radii, best_metrics

    def visualize_tree(self, tree=None, root=None, radii=None, show_radii=True):
        """
        Visualize the reconstructed tree
        """
        if tree is None:
            tree = self.tree_structure['tree']
            root = self.tree_structure['root']
            radii = self.tree_structure['radii']

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Draw edges
        for u, v in tree.edges():
            pos_u = self.points[u]
            pos_v = self.points[v]

            if show_radii and radii:
                # Line width proportional to radius
                radius = radii[v]
                # Scale for visibility
                linewidth = max(0.5, min(5, radius * 10))
                color = plt.cm.RdYlBu_r(radius / max(radii.values()))
            else:
                linewidth = 1
                color = 'blue'

            ax.plot([pos_u[0], pos_v[0]],
                    [pos_u[1], pos_v[1]],
                    [pos_u[2], pos_v[2]],
                    'b-', linewidth=linewidth, color=color, alpha=0.6)

        # Highlight root
        root_pos = self.points[root]
        ax.scatter(*root_pos, c='red', s=100, marker='o', label='Root')

        # Highlight terminals
        terminals = [n for n in tree.nodes() if tree.out_degree(n) == 0]
        terminal_pos = self.points[terminals]
        ax.scatter(terminal_pos[:, 0], terminal_pos[:, 1], terminal_pos[:, 2],
                   c='green', s=20, marker='^', label='Terminals', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Reconstructed Vascular Tree')

        plt.tight_layout()
        return fig, ax

    def export_tree(self, filename, tree=None, radii=None):
        """
        Export tree to file format (e.g., for visualization or further processing)
        """
        if tree is None:
            tree = self.tree_structure['tree']
            radii = self.tree_structure['radii']

        with open(filename, 'w') as f:
            f.write("# Vascular Tree Structure\n")
            f.write("# Format: node_id parent_id x y z radius\n")

            root = self.tree_structure['root']
            f.write(
                f"{root} -1 {' '.join(map(str, self.points[root]))} {radii[root]:.6f}\n")

            for node in nx.dfs_preorder_nodes(tree, root):
                if node == root:
                    continue
                parent = list(tree.predecessors(node))[0]
                pos = self.points[node]
                radius = radii[node]
                f.write(
                    f"{node} {parent} {' '.join(map(str, pos))} {radius:.6f}\n")

        print(f"Tree exported to {filename}")

    def export_tree_for_viewer(self, filename, tree=None, radii=None):
        """
        Export tree to JSON format compatible with the multi-volume viewer

        Format expected by viewer:
        [
        {
            "start": [x1, y1, z1],
            "end": [x2, y2, z2],
            "radius": 1.2,
            "Q": 5
        },
        ...
        ]

        Parameters:
        -----------
        filename : str
            Output JSON filename
        tree : networkx.DiGraph (optional)
            Tree to export (uses self.tree_structure if None)
        radii : dict (optional)
            Radii for each node (uses self.tree_structure if None)
        """
        import json

        if tree is None:
            tree = self.tree_structure['tree']
            radii = self.tree_structure['radii']
            root = self.tree_structure['root']

        branches = []

        # Iterate through all edges in the tree
        for parent, child in tree.edges():
            # Get positions
            start_pos = self.points[parent].tolist()
            end_pos = self.points[child].tolist()

            # Get radius (use child's radius for the segment)
            radius = float(radii[child])

            # Compute flow Q (proportional to r³ from Equation 10)
            # Scale to reasonable values (relative flow)
            Q = float(radius ** 3 * 100)  # Scale factor for visualization

            branch = {
                "start": start_pos,
                "end": end_pos,
                "radius": radius,
                "Q": Q
            }

            branches.append(branch)

        # Write to JSON file
        with open(filename, 'w') as f:
            json.dump(branches, f, indent=2)

        print(f"Tree exported to {filename} ({len(branches)} branches)")
        print(
            f"Radius range: [{min(b['radius'] for b in branches):.4f}, {max(b['radius'] for b in branches):.4f}]")


class MultiTreeReconstruction:
    """
    Reconstruct multiple independent vascular trees from a single point cloud
    """

    def __init__(self, skeleton_points, n_trees=3, gamma=3.0, mu=3.6e-3,
                 Q_perf=0.125, P_out=60, P_in=100):
        """
        Parameters:
        -----------
        skeleton_points : np.array, shape (N, 3)
            3D coordinates of skeleton points
        n_trees : int
            Expected number of separate trees to extract
        gamma, mu, Q_perf, P_out, P_in : float
            CCO physiological parameters (same as VascularTreeReconstruction)
        """
        self.original_points = np.array(skeleton_points)
        self.n_trees = n_trees
        self.gamma = gamma
        self.mu = mu
        self.Q_perf = Q_perf
        self.P_out = P_out
        self.P_in = P_in

        self.trees = []  # List of reconstructed trees
        self.remaining_points = self.original_points.copy()
        self.point_to_tree_mapping = {}  # Maps original point index to tree index

    def reconstruct_multiple_trees(self, k_neighbors_initial=5, k_neighbors_optimization=10,
                                   min_tree_size=50, max_iterations=10, methods=["highest_z", "highest_z", "lowest_z"]):
        """
        Extract multiple trees iteratively

        Parameters:
        -----------
        k_neighbors_initial : int
            Low k for initial sparse connectivity (helps separate trees)
        k_neighbors_optimization : int
            Higher k for optimizing individual trees
        min_tree_size : int
            Minimum number of points required for a valid tree
        max_iterations : int
            Maximum number of trees to extract

        Returns:
        --------
        trees : list of dict
            List of tree structures, each containing:
            - 'tree': networkx.DiGraph
            - 'root': int (root node in original point indexing)
            - 'radii': dict
            - 'metrics': dict
            - 'points': np.array (points for this tree)
            - 'point_indices': list (indices in original point cloud)
        """
        print(
            f"Starting multi-tree reconstruction from {len(self.original_points)} points")
        print(f"Target: {self.n_trees} trees\n")

        iteration = 0

        while len(self.remaining_points) >= min_tree_size and iteration < max_iterations:
            iteration += 1
            print(f"{'='*60}")
            print(
                f"ITERATION {iteration}: {len(self.remaining_points)} points remaining")
            print(f"{'='*60}\n")

            root_method = methods[(iteration - 1) % len(methods)]
            print(f"Using root candidate method: {root_method}")

            # Extract one tree from remaining points
            tree_data = self._extract_single_tree(
                k_neighbors_initial=k_neighbors_initial,
                k_neighbors_optimization=k_neighbors_optimization,
                min_tree_size=min_tree_size,
                method=root_method
            )

            if tree_data is None:
                print("No more valid trees found. Stopping.")
                break

            self.trees.append(tree_data)

            # Remove points belonging to this tree
            self._remove_tree_points(tree_data)

            print(
                f"\n✓ Tree {iteration} extracted: {len(tree_data['point_indices'])} points")
            print(f"  Root at: {tree_data['points'][tree_data['root']]}")
            print(
                f"  Terminals: {len([n for n in tree_data['tree'].nodes() if tree_data['tree'].out_degree(n) == 0])}")
            print(
                f"  Total volume: {tree_data['metrics']['total_volume']:.2f} mm³")
            print(f"  Remaining points: {len(self.remaining_points)}\n")

            # Stop if we've extracted enough trees
            if len(self.trees) >= self.n_trees:
                print(
                    f"Extracted target number of trees ({self.n_trees}). Stopping.")
                break

        print(f"\n{'='*60}")
        print(f"FINAL RESULT: {len(self.trees)} trees extracted")
        print(f"{'='*60}")

        for i, tree_data in enumerate(self.trees):
            print(f"Tree {i+1}: {len(tree_data['point_indices'])} points, "
                  f"volume={tree_data['metrics']['total_volume']:.2f} mm³")

        if len(self.remaining_points) > 0:
            print(
                f"\nWarning: {len(self.remaining_points)} points not assigned to any tree")

        return self.trees

    def _extract_single_tree(self, k_neighbors_initial, k_neighbors_optimization, min_tree_size, method):
        """
        Extract one tree from remaining points using sparse connectivity
        """
        if len(self.remaining_points) < min_tree_size:
            return None

        # Step 1: Build sparse graph to identify connected components
        print(f"Step 1: Building sparse graph (k={k_neighbors_initial})...")
        sparse_graph = self._build_sparse_graph(
            self.remaining_points, k_neighbors_initial)

        if sparse_graph.number_of_edges() == 0:
            print("  No edges in sparse graph. Cannot extract tree.")
            return None

        # Step 2: Find largest connected component
        print(f"Step 2: Finding connected components...")
        components = list(nx.connected_components(sparse_graph))
        print(f"  Found {len(components)} components")

        if len(components) == 0:
            return None

        # Get largest component
        largest_component = max(components, key=len)
        print(f"  Largest component: {len(largest_component)} nodes")

        if len(largest_component) < min_tree_size:
            print(f"  Component too small (< {min_tree_size}). Skipping.")
            return None

        # Step 3: Extract points for this component
        component_indices = sorted(list(largest_component))
        component_points = self.remaining_points[component_indices]

        print(
            f"Step 3: Extracting subgraph with {len(component_points)} points...")

        # Step 4: Optimize this tree using standard CCO with higher k
        print(f"Step 4: Optimizing tree (k={k_neighbors_optimization})...")
        reconstructor = VascularTreeReconstruction(
            component_points,
            gamma=self.gamma,
            mu=self.mu,
            Q_perf=self.Q_perf,
            P_out=self.P_out,
            P_in=self.P_in
        )

        try:
            tree, root, radii, metrics = reconstructor.reconstruct(
                k_neighbors=k_neighbors_optimization,
                n_candidates=5,
                optimize_bifurcations=True,
                method=method
            )
        except Exception as e:
            print(f"  Failed to reconstruct tree: {e}")
            return None

        # Step 5: Map back to original point indices
        # component_indices maps from component local indices to remaining_points indices
        # We need to map to original point cloud indices

        # Find original indices for these remaining points
        original_indices = self._get_original_indices(component_indices)

        tree_data = {
            'tree': tree,
            'root': root,  # Root in local component indexing
            'radii': radii,
            'metrics': metrics,
            'points': component_points,
            'point_indices': original_indices,  # Indices in original point cloud
            'reconstructor': reconstructor
        }

        return tree_data

    def _build_sparse_graph(self, points, k_neighbors):
        """
        Build a sparse k-NN graph (lower k helps separate disconnected trees)
        """
        from scipy.spatial import cKDTree

        n_points = len(points)
        tree = cKDTree(points)

        # Create graph
        G = nx.Graph()

        # Add all nodes
        for i in range(n_points):
            G.add_node(i)

        # Add k-NN edges
        for i in range(n_points):
            dists, indices = tree.query(
                points[i], k=min(k_neighbors+1, n_points))
            for j, idx in enumerate(indices[1:]):  # Skip self
                if dists[j+1] < np.inf:
                    G.add_edge(i, idx, weight=dists[j+1])

        return G

    def _get_original_indices(self, component_indices):
        """
        Map component indices (in remaining_points) back to original point cloud indices
        """
        # Build reverse mapping: remaining_points → original_points
        original_indices = []

        for comp_idx in component_indices:
            remaining_point = self.remaining_points[comp_idx]

            # Find this point in original cloud
            # Use exact match (or nearest neighbor if points have been modified)
            distances = np.linalg.norm(
                self.original_points - remaining_point, axis=1)
            orig_idx = np.argmin(distances)

            # Verify it's a close match
            if distances[orig_idx] < 0.01:  # Should be exact or very close
                original_indices.append(orig_idx)
            else:
                print(
                    f"Warning: Could not find exact match for point {comp_idx}")
                original_indices.append(orig_idx)  # Use closest anyway

        return original_indices

    def _remove_tree_points(self, tree_data):
        """
        Remove points belonging to extracted tree from remaining points
        """
        # Get indices of points to remove (in remaining_points indexing)
        points_to_remove = tree_data['points']

        # Find which indices in remaining_points to keep
        keep_mask = np.ones(len(self.remaining_points), dtype=bool)

        for i, point in enumerate(self.remaining_points):
            # Check if this point is in the extracted tree
            distances = np.linalg.norm(points_to_remove - point, axis=1)
            if np.min(distances) < 0.01:  # Point is in tree
                keep_mask[i] = False

        # Update remaining points
        self.remaining_points = self.remaining_points[keep_mask]

    def visualize_all_trees(self, show_remaining=True):
        """
        Visualize all extracted trees in a single 3D plot
        """
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Color palette for different trees
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.trees)))

        for tree_idx, tree_data in enumerate(self.trees):
            tree = tree_data['tree']
            points = tree_data['points']
            radii = tree_data['radii']
            root = tree_data['root']
            color = colors[tree_idx]

            # Draw edges
            for u, v in tree.edges():
                pos_u = points[u]
                pos_v = points[v]
                radius = radii[v]
                linewidth = max(0.5, min(5, radius * 10))

                ax.plot([pos_u[0], pos_v[0]],
                        [pos_u[1], pos_v[1]],
                        [pos_u[2], pos_v[2]],
                        color=color, linewidth=linewidth, alpha=0.7)

            # Highlight root
            root_pos = points[root]
            ax.scatter(*root_pos, c=[color], s=200, marker='o',
                       edgecolors='black', linewidths=2,
                       label=f'Tree {tree_idx+1} root')

            # Show terminals
            terminals = [n for n in tree.nodes() if tree.out_degree(n) == 0]
            terminal_pos = points[terminals]
            if len(terminal_pos) > 0:
                ax.scatter(terminal_pos[:, 0], terminal_pos[:, 1], terminal_pos[:, 2],
                           c=[color], s=30, marker='^', alpha=0.5)

        # Show remaining unassigned points
        if show_remaining and len(self.remaining_points) > 0:
            ax.scatter(self.remaining_points[:, 0],
                       self.remaining_points[:, 1],
                       self.remaining_points[:, 2],
                       c='gray', s=5, alpha=0.3, label='Unassigned points')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f'Multi-Tree Reconstruction ({len(self.trees)} trees)')

        plt.tight_layout()
        return fig, ax

    def export_all_trees(self, base_filename):
        """
        Export all trees to separate JSON files for viewer

        Parameters:
        -----------
        base_filename : str
            Base name for output files (e.g., "hepatic_vein")
            Will create: hepatic_vein_tree1.json, hepatic_vein_tree2.json, etc.
        """
        import json

        for tree_idx, tree_data in enumerate(self.trees):
            filename = f"{base_filename}_tree{tree_idx+1}.json"

            tree = tree_data['tree']
            points = tree_data['points']
            radii = tree_data['radii']

            branches = []

            for parent, child in tree.edges():
                start_pos = points[parent].tolist()
                end_pos = points[child].tolist()
                radius = float(radii[child])
                Q = float(radius ** 3 * 100)

                branch = {
                    "start": start_pos,
                    "end": end_pos,
                    "radius": radius,
                    "Q": Q
                }
                branches.append(branch)

            with open(filename, 'w') as f:
                json.dump(branches, f, indent=2)

            print(
                f"Tree {tree_idx+1} exported to {filename} ({len(branches)} branches)")

    def get_tree_statistics(self):
        """
        Print summary statistics for all trees
        """
        print(f"\n{'='*60}")
        print(f"MULTI-TREE STATISTICS")
        print(f"{'='*60}\n")

        total_points = sum(len(t['point_indices']) for t in self.trees)
        total_volume = sum(t['metrics']['total_volume'] for t in self.trees)

        print(f"Total trees: {len(self.trees)}")
        print(
            f"Total points assigned: {total_points} / {len(self.original_points)}")
        print(f"Total volume: {total_volume:.2f} mm³")
        print(f"Unassigned points: {len(self.remaining_points)}\n")

        for i, tree_data in enumerate(self.trees):
            print(f"Tree {i+1}:")
            print(f"  Points: {len(tree_data['point_indices'])}")
            print(f"  Root: {tree_data['points'][tree_data['root']]}")

            tree = tree_data['tree']
            terminals = [n for n in tree.nodes() if tree.out_degree(n) == 0]
            bifurcations = [n for n in tree.nodes()
                            if tree.out_degree(n) == 2 and tree.in_degree(n) == 1]

            print(f"  Terminals: {len(terminals)}")
            print(f"  Bifurcations: {len(bifurcations)}")
            print(f"  Volume: {tree_data['metrics']['total_volume']:.2f} mm³")
            print(
                f"  Murray error: {tree_data['metrics']['murray_error_mean']:.4f}")

            radii_values = list(tree_data['radii'].values())
            print(
                f"  Radius range: [{min(radii_values):.3f}, {max(radii_values):.3f}] mm")
            print()


# Helper functions
# TODO create package and use from there instead of copy-pasting

def load_thinning(labels):
    skeleton = skeletonize(labels)
    skeleton_points = np.array(np.where(skeleton)).T
    return skeleton_points


def save_skeleton(skeleton_points, filename):
    # Ensure a pure Python list is saved (JSON cannot serialize numpy arrays)
    to_save = np.array(skeleton_points).tolist()
    with open(filename, 'w') as f:
        json.dump(to_save, f)
    print(f"Skeleton saved to {filename}")
