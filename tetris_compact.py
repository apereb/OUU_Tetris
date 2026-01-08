import numpy as np
from collections import deque
import time


class CompactTetris:
    def __init__(self, H):
        self.H = H
        self.W = 4
        
    def state_to_idx(self, heights, piece):
        idx = 0
        for i, h in enumerate(heights):
            idx += h * ((self.H + 1) ** i)
        idx = idx * 2 + (piece - 1)
        return idx
    
    def idx_to_state(self, idx):
        piece = (idx % 2) + 1
        idx = idx // 2
        
        heights = []
        for i in range(4):
            h = idx % (self.H + 1)
            heights.append(h)
            idx = idx // (self.H + 1)
        
        return tuple(heights), piece
    
    def heights_to_grid(self, heights):
        grid = np.zeros((self.H, 4), dtype=int)
        for col, h in enumerate(heights):
            for row in range(h):
                grid[row, col] = 1
        return grid
    
    def is_valid_placement(self, heights, coords):
        for row, col in coords:
            if col < 0 or col >= 4 or row < 0 or row >= self.H:
                return False
            if row < heights[col]:
                return False
            if row > heights[col]:
                supported = False
                if row == 0:
                    supported = True
                else:
                    for other_row, other_col in coords:
                        if other_col == col and other_row == row - 1:
                            supported = True
                            break
                    if row == heights[col]:
                        supported = True
                
                if not supported and row > heights[col]:
                    return False
        
        return True
    
    def apply_placement(self, heights, coords):
        new_heights = list(heights)
        for row, col in coords:
            new_heights[col] = max(new_heights[col], row + 1)
        min_old = min(heights)
        min_new = min(new_heights)
        
        clearable_lines = []
        for h in range(self.H):
            if all(height > h for height in new_heights):
                clearable_lines.append(h)
        
        old_clearable_count = sum(1 for h in range(self.H) if all(height > h for height in heights))
        new_clearable_count = len(clearable_lines)
        
        lines_cleared = new_clearable_count - old_clearable_count
        lines_cleared = max(0, lines_cleared)
        
        if lines_cleared > 0:
            new_heights = [h - lines_cleared for h in new_heights]
        
        reward = {0: 0, 1: 1, 2: 3, 3: 6}.get(lines_cleared, 0)
        
        return tuple(new_heights), reward
    
    def get_possible_placements(self, heights, piece_type):
        results = []
        seen_states = set()
        
        if piece_type == 1:
            rotations = [
                [[0, 0], [0, 1], [0, 2]],
                [[0, 0], [1, 0], [2, 0]]
            ]
        else:
            rotations = [
                [[0, 0], [0, 1], [1, 1]], 
                [[0, 1], [1, 0], [1, 1]],
                [[0, 0], [1, 0], [1, 1]],
                [[0, 0], [0, 1], [1, 0]]
            ]
        
        for rotation in rotations:
            min_col = min(c for r, c in rotation)
            max_col = max(c for r, c in rotation)
            width = max_col - min_col + 1
            
            for x_offset in range(5 - width):
                adjusted = [[r, c - min_col + x_offset] for r, c in rotation]
                
                max_base_height = max(heights[col] for row, col in adjusted)
                
                y_offset = max_base_height
                placed_coords = [[row + y_offset, col] for row, col in adjusted]
                
                if any(row >= self.H for row, col in placed_coords):
                    continue
                
                new_heights, reward = self.apply_placement(heights, placed_coords)
                
                if all(h <= self.H for h in new_heights):
                    if new_heights not in seen_states:
                        seen_states.add(new_heights)
                        results.append((new_heights, reward))
        
        return results
    
    def enumerate_reachable_states(self):
        initial_heights = (0, 0, 0, 0)
        queue = deque([
            (initial_heights, 1),
            (initial_heights, 2)
        ])
        
        seen = set()
        all_states = []
        global_to_vi = {}
        vi_to_global = {}
        
        idx = 0
        
        while queue:
            heights, piece = queue.popleft()
            state_idx = self.state_to_idx(heights, piece)
            
            if state_idx in seen:
                continue
            
            seen.add(state_idx)
            all_states.append(state_idx)
            global_to_vi[state_idx] = idx
            vi_to_global[idx] = state_idx
            idx += 1
            
            placements = self.get_possible_placements(heights, piece)
            
            for new_heights, reward in placements:
                for next_piece in [1, 2]:
                    next_idx = self.state_to_idx(new_heights, next_piece)
                    if next_idx not in seen:
                        queue.append((new_heights, next_piece))
        
        return all_states, global_to_vi, vi_to_global
    
    def build_transition_model(self):
        all_states, global_to_vi, vi_to_global = self.enumerate_reachable_states()
        R = len(all_states)
        terminal_idx = R
        
        actions = [[] for _ in range(R + 1)]
        
        for vi_idx in range(R):
            global_idx = vi_to_global[vi_idx]
            heights, piece = self.idx_to_state(global_idx)
            
            placements = self.get_possible_placements(heights, piece)
            if len(placements) == 0:
                actions[vi_idx].append((0, [(1.0, terminal_idx)]))
            
            for new_heights, reward in placements:
                if any(h > self.H for h in new_heights):
                    actions[vi_idx].append((reward, [(1.0, terminal_idx)]))
                else:
                    transitions = []
                    for next_piece in [1, 2]:
                        next_global = self.state_to_idx(new_heights, next_piece)
                        if next_global in global_to_vi:
                            next_vi = global_to_vi[next_global]
                            transitions.append((0.5, next_vi))
                    
                    if transitions:
                        actions[vi_idx].append((reward, transitions))
        
        return actions, all_states, global_to_vi, vi_to_global
    
    def value_iteration(self, gamma=0.99, epsilon=1e-9, max_iter=10000):
        actions, all_states, global_to_vi, vi_to_global = self.build_transition_model()
        R = len(all_states)
        
        V = np.zeros(R + 1)
        policy = [-1] * R
        
        for iteration in range(max_iter):
            V_old = V.copy()
            
            for i in range(R):
                if len(actions[i]) == 0:
                    continue
                
                best_value = -np.inf
                best_action = -1
                
                for action_idx, (reward, transitions) in enumerate(actions[i]):
                    q_value = reward + gamma * sum(prob * V_old[next_vi] 
                                                   for prob, next_vi in transitions)
                    
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action_idx
                
                V[i] = best_value
                policy[i] = best_action
            
            if np.max(np.abs(V - V_old)) < epsilon:
                print(f"Converged in {iteration + 1} iterations")
                return V, policy, iteration + 1, all_states, global_to_vi, vi_to_global
        
        print(f"Did not converge after {max_iter} iterations")
        return V, policy, max_iter, all_states, global_to_vi, vi_to_global
    
    def value_iteration_adversarial(self, gamma=0.99, epsilon=1e-9, max_iter=10000):
        actions, all_states, global_to_vi, vi_to_global = self.build_transition_model()
        R = len(all_states)
        
        V = np.zeros(R + 1)
        policy = [-1] * R
        
        for iteration in range(max_iter):
            V_old = V.copy()
            
            for i in range(R):
                if len(actions[i]) == 0:
                    continue
                
                best_value = -np.inf
                best_action = -1
                
                for action_idx, (reward, transitions) in enumerate(actions[i]):
                    if len(transitions) > 0:
                        worst_next = min(V_old[next_vi] for prob, next_vi in transitions)
                    else:
                        worst_next = 0
                    
                    q_value = reward + gamma * worst_next
                    
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action_idx
                
                V[i] = best_value
                policy[i] = best_action
            
            if np.max(np.abs(V - V_old)) < epsilon:
                print(f"Converged in {iteration + 1} iterations")
                return V, policy, iteration + 1, all_states, global_to_vi, vi_to_global
        
        print(f"Did not converge after {max_iter} iterations")
        return V, policy, max_iter, all_states, global_to_vi, vi_to_global


def compare_methods(H=4):
    
    game = CompactTetris(H)
    start = time.time()
    all_states, global_to_vi, vi_to_global = game.enumerate_reachable_states()
    enum_time = time.time() - start
    
    print(f"Number of reachable states: {len(all_states)}")
    print(f"State space size: {(H+1)**4 * 2:,} (theoretical max)")
    print(f"Reachable ratio: {len(all_states) / ((H+1)**4 * 2):.2%}")
    print(f"Enumeration time: {enum_time:.4f} seconds")
    
    gamma_values = [0.90, 0.95, 0.99, 0.999]
    
    print("STANDARD MDP")
    print(f"\n{'Gamma':<8} {'V*(p_0)':<12} {'V*(p_1)':<12} {'Iterations':<12} {'Time(s)':<12}")
    results_mdp = []
    
    for gamma in gamma_values:
        start = time.time()
        V, policy, iters, states, g2vi, vi2g = game.value_iteration(gamma=gamma)
        elapsed = time.time() - start
        
        init_heights = (0, 0, 0, 0)
        idx_p1 = game.state_to_idx(init_heights, 1)
        idx_p2 = game.state_to_idx(init_heights, 2)
        
        vi_p1 = g2vi[idx_p1]
        vi_p2 = g2vi[idx_p2]
        
        print(f"{gamma:<8.3f} {V[vi_p1]:<12.2f} {V[vi_p2]:<12.2f} {iters:<12} {elapsed:<12.4f}")
        
        results_mdp.append({
            'gamma': gamma,
            'V_p0': V[vi_p1],
            'V_p1': V[vi_p2],
            'iters': iters,
            'time': elapsed
        })
    
    print("ADVERSARIAL GAME")
    print(f"\n{'Gamma':<8} {'V*(p_0)':<12} {'V*(p_1)':<12} {'Iterations':<12} {'Time(s)':<12}")
    
    results_adv = []
    
    for gamma in gamma_values:
        start = time.time()
        V, policy, iters, states, g2vi, vi2g = game.value_iteration_adversarial(gamma=gamma)
        elapsed = time.time() - start
        
        init_heights = (0, 0, 0, 0)
        idx_p1 = game.state_to_idx(init_heights, 1)
        idx_p2 = game.state_to_idx(init_heights, 2)
        
        vi_p1 = g2vi[idx_p1]
        vi_p2 = g2vi[idx_p2]
        
        print(f"{gamma:<8.3f} {V[vi_p1]:<12.2f} {V[vi_p2]:<12.2f} {iters:<12} {elapsed:<12.4f}")
        
        results_adv.append({
            'gamma': gamma,
            'V_p0': V[vi_p1],
            'V_p1': V[vi_p2],
            'iters': iters,
            'time': elapsed
        })
    
    return results_mdp, results_adv


if __name__ == "__main__":
    print("TESTING COMPACT REPRESENTATION")
    
    print("\nTesting H=4")
    results_4_mdp, results_4_adv = compare_methods(H=4)
    
    print("\n\nTesting H=6")
    results_6_mdp, results_6_adv = compare_methods(H=6)
    
    print("\n\nTesting H=8")
    results_8_mdp, results_8_adv = compare_methods(H=8)
    
    print("SUMMARY COMPARISON")
    
    print("\nFor γ=0.99:")
    print(f"\n{'Height':<8} {'States':<12} {'MDP V*':<12} {'Adv V*':<12} {'Ratio':<12}")
    
    for H, res_mdp, res_adv in [(4, results_4_mdp, results_4_adv),
                                  (6, results_6_mdp, results_6_adv),
                                  (8, results_8_mdp, results_8_adv)]:
        game_temp = CompactTetris(H)
        states, _, _ = game_temp.enumerate_reachable_states()
        
        mdp_99 = [r for r in res_mdp if r['gamma'] == 0.99][0]
        adv_99 = [r for r in res_adv if r['gamma'] == 0.99][0]
        
        ratio = adv_99['V_p0'] / mdp_99['V_p0'] * 100
        
        print(f"{H:<8} {len(states):<12,} {mdp_99['V_p0']:<12.2f} {adv_99['V_p1']:<12.2f} {ratio:<12.1f}%")
