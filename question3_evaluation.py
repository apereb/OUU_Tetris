import numpy as np
import sys
sys.path.append('/home/claude')
from tetris import Jeu
import time
from collections import defaultdict


class AdversarySimulator:
    def __init__(self, game, V_mdp=None, V_adv=None):
        self.game = game
        self.V_mdp = V_mdp
        self.V_adv = V_adv
    
    def uniform_random(self, grid, global_to_vi=None):
        return np.random.choice([1, 2])
    
    def adversarial_worst(self, grid, global_to_vi):
        if self.V_adv is None or global_to_vi is None:
            return np.random.choice([1, 2])
        
        idx_p1 = self.game.state_to_idx(grid, 1)
        idx_p2 = self.game.state_to_idx(grid, 2)
        
        if idx_p1 in global_to_vi and idx_p2 in global_to_vi:
            vi_p1 = global_to_vi[idx_p1]
            vi_p2 = global_to_vi[idx_p2]
            
            v_p1 = self.V_adv[vi_p1]
            v_p2 = self.V_adv[vi_p2]
            
            return 1 if v_p1 <= v_p2 else 2
        
        return np.random.choice([1, 2])
    
    def adversarial_best(self, grid, global_to_vi):
        if self.V_mdp is None or global_to_vi is None:
            return np.random.choice([1, 2])
        
        idx_p1 = self.game.state_to_idx(grid, 1)
        idx_p2 = self.game.state_to_idx(grid, 2)
        
        if idx_p1 in global_to_vi and idx_p2 in global_to_vi:
            vi_p1 = global_to_vi[idx_p1]
            vi_p2 = global_to_vi[idx_p2]
            
            v_p1 = self.V_mdp[vi_p1]
            v_p2 = self.V_mdp[vi_p2]
            return 1 if v_p1 >= v_p2 else 2
        
        return np.random.choice([1, 2])
    
    def biased_70_30(self, grid, global_to_vi):
        worst_piece = self.adversarial_worst(grid, global_to_vi)
        best_piece = 3 - worst_piece 
        
        return np.random.choice([worst_piece, best_piece], p=[0.7, 0.3])
    
    def biased_30_70(self, grid, global_to_vi):
        worst_piece = self.adversarial_worst(grid, global_to_vi)
        best_piece = 3 - worst_piece
        
        return np.random.choice([worst_piece, best_piece], p=[0.3, 0.7])
    
    def alternating(self, grid, global_to_vi=None):
        if not hasattr(self, 'last_piece'):
            self.last_piece = 1
        self.last_piece = 3 - self.last_piece
        return self.last_piece
    
    def switching(self, grid, global_to_vi=None):
        if not hasattr(self, 'move_count'):
            self.move_count = 0
        
        self.move_count += 1
        
        if (self.move_count // 10) % 3 == 0:
            return self.uniform_random(grid, global_to_vi)
        elif (self.move_count // 10) % 3 == 1:
            return self.adversarial_worst(grid, global_to_vi)
        else:
            return self.alternating(grid, global_to_vi)
    
    def markov_chain(self, grid, global_to_vi=None):
        if not hasattr(self, 'current_piece_markov'):
            self.current_piece_markov = 1
        
        transition = [[0.7, 0.3], [0.4, 0.6]]
        
        if self.current_piece_markov == 1:
            probs = transition[0]
        else:
            probs = transition[1]
        
        self.current_piece_markov = np.random.choice([1, 2], p=probs)
        return self.current_piece_markov


class MixedStrategyPolicy:
    def __init__(self, policies, mixing_weights=None):
        self.policies = policies
        if mixing_weights is None:
            self.mixing_weights = np.ones(len(policies)) / len(policies)
        else:
            self.mixing_weights = np.array(mixing_weights)
            self.mixing_weights = self.mixing_weights / self.mixing_weights.sum()
    
    def get_action(self, state_vi_idx):
        selected_policy_idx = np.random.choice(len(self.policies), p=self.mixing_weights)
        selected_policy = self.policies[selected_policy_idx]
        
        if state_vi_idx < len(selected_policy):
            return selected_policy[state_vi_idx]
        return -1


class GameSimulator:
    def __init__(self, H=4):
        self.H = H
        self.game = Jeu(H)
    
    def play_game(self, policy, actions, global_to_vi, vi_to_global, adversary_func, max_moves=1000, is_mixed=False):
        grid = np.zeros((self.H, 4), dtype=int)
        score = 0
        moves = 0
        
        current_piece = np.random.choice([1, 2])
        
        for move in range(max_moves):
            moves += 1
            
            state_idx = self.game.state_to_idx(grid, current_piece)
            
            if state_idx not in global_to_vi:
                return score, moves, "unknown_state"
            
            vi_idx = global_to_vi[state_idx]
            
            if vi_idx >= len(actions) or len(actions[vi_idx]) == 0:
                return score, moves, "no_valid_actions"
            
            if is_mixed:
                action_idx = policy.get_action(vi_idx)
            else:
                action_idx = policy[vi_idx]
            
            if action_idx < 0 or action_idx >= len(actions[vi_idx]):
                return score, moves, "invalid_policy"
            
            action_info = actions[vi_idx][action_idx]
            reward = action_info[0]
            transitions = action_info[1]
            
            terminal_idx = len(global_to_vi)
            if any(next_vi == terminal_idx for prob, next_vi in transitions):
                return score, moves, "height_exceeded"
            
            if len(transitions) == 0:
                return score, moves, "terminal_state"
            
            next_vi_sample = transitions[0][1]
            
            if next_vi_sample >= len(vi_to_global):
                return score, moves, "terminal_state"
            
            next_global_sample = vi_to_global[next_vi_sample]
            new_grid, _ = self.game.idx_to_state(next_global_sample)
            
            grid = new_grid
            score += reward
            
            current_piece = adversary_func(grid, global_to_vi)
        
        return score, moves, "max_moves_reached"
    
    def evaluate_policy(self, policy, actions, global_to_vi, vi_to_global, 
                       adversary_func, adversary_name, num_games=100, is_mixed=False):
        scores = []
        move_counts = []
        reasons = defaultdict(int)
        
        for game_num in range(num_games):
            score, moves, reason = self.play_game(
                policy, actions, global_to_vi, vi_to_global, adversary_func, is_mixed=is_mixed
            )
            scores.append(score)
            move_counts.append(moves)
            reasons[reason] += 1
        
        return {
            'adversary': adversary_name,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'median_score': np.median(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'mean_moves': np.mean(move_counts),
            'std_moves': np.std(move_counts),
            'game_over_reasons': dict(reasons)
        }


def main():
    
    H = 4
    game = Jeu(H)
    
    print(f"\nGrid size: {H}×4")
    
    print("\n1. Computing MDP policy (uniform random adversary)...")
    start = time.time()
    V_mdp, policy_mdp = game.Value_iteration(gamma=0.99)
    time_mdp = time.time() - start
    print(f"   Done in {time_mdp:.2f}s")
    
    print("\n2. Computing Adversarial policy (worst-case adversary)...")
    start = time.time()
    V_adv, policy_adv = game.Value_iteration_adversarial(gamma=0.99)
    time_adv = time.time() - start
    print(f"   Done in {time_adv:.2f}s")
    
    print("\n3. Computing additional policies for mixed strategy...")
    start = time.time()
    V_mdp_90, policy_mdp_90 = game.Value_iteration(gamma=0.90)
    V_mdp_95, policy_mdp_95 = game.Value_iteration(gamma=0.95)
    time_extra = time.time() - start
    print(f"   Done in {time_extra:.2f}s")
    
    print("\n4. Creating mixed strategy policies...")
    mixed_uniform = MixedStrategyPolicy(
        [policy_mdp_90, policy_mdp_95, policy_mdp, policy_adv],
        [0.25, 0.25, 0.25, 0.25]
    )
    
    mixed_conservative = MixedStrategyPolicy(
        [policy_mdp, policy_adv],
        [0.3, 0.7]
    )
    
    mixed_optimistic = MixedStrategyPolicy(
        [policy_mdp, policy_adv],
        [0.7, 0.3]
    )
    
    mixed_balanced = MixedStrategyPolicy(
        [policy_mdp, policy_adv],
        [0.5, 0.5]
    )
    
    print("   Created 4 mixed strategy variants")
    
    all_states, global_to_vi, vi_to_global = game.all_state()
    actions = game.rewards_transition()
    print(f"   Reachable states: {len(all_states)}")
    
    adversary_sim = AdversarySimulator(game, V_mdp, V_adv)
    game_sim = GameSimulator(H)
    
    adversaries = [
        ('Uniform Random', adversary_sim.uniform_random),
        ('Adversarial (Worst)', adversary_sim.adversarial_worst),
        ('Adversarial (Best)', adversary_sim.adversarial_best),
        ('Biased 70-30 (Worst)', adversary_sim.biased_70_30),
        ('Biased 30-70 (Best)', adversary_sim.biased_30_70),
        ('Alternating', adversary_sim.alternating),
        ('Switching (Non-stationary)', adversary_sim.switching),
        ('Markov Chain', adversary_sim.markov_chain),
    ]
    
    num_games = 100
    
    print("RESULTS")
    
    results_mdp = []
    results_adv = []
    results_mixed_uniform = []
    results_mixed_conservative = []
    results_mixed_optimistic = []
    results_mixed_balanced = []
    
    for adv_name, adv_func in adversaries:
        print(f"Testing against: {adv_name}")
        if hasattr(adversary_sim, 'move_count'):
            delattr(adversary_sim, 'move_count')
        if hasattr(adversary_sim, 'last_piece'):
            delattr(adversary_sim, 'last_piece')
        if hasattr(adversary_sim, 'current_piece_markov'):
            delattr(adversary_sim, 'current_piece_markov')
        
        result_mdp = game_sim.evaluate_policy(
            policy_mdp, actions, global_to_vi, vi_to_global,
            adv_func, adv_name, num_games
        )
        results_mdp.append(result_mdp)
        
        if hasattr(adversary_sim, 'move_count'):
            delattr(adversary_sim, 'move_count')
        if hasattr(adversary_sim, 'last_piece'):
            delattr(adversary_sim, 'last_piece')
        if hasattr(adversary_sim, 'current_piece_markov'):
            delattr(adversary_sim, 'current_piece_markov')
        
        result_adv = game_sim.evaluate_policy(
            policy_adv, actions, global_to_vi, vi_to_global,
            adv_func, adv_name, num_games
        )
        results_adv.append(result_adv)
        
        for mixed_policy, results_list, name in [
            (mixed_uniform, results_mixed_uniform, "Uniform"),
            (mixed_conservative, results_mixed_conservative, "Conservative"),
            (mixed_optimistic, results_mixed_optimistic, "Optimistic"),
            (mixed_balanced, results_mixed_balanced, "Balanced")
        ]:
            if hasattr(adversary_sim, 'move_count'):
                delattr(adversary_sim, 'move_count')
            if hasattr(adversary_sim, 'last_piece'):
                delattr(adversary_sim, 'last_piece')
            if hasattr(adversary_sim, 'current_piece_markov'):
                delattr(adversary_sim, 'current_piece_markov')
            
            result = game_sim.evaluate_policy(
                mixed_policy, actions, global_to_vi, vi_to_global,
                adv_func, adv_name, num_games, is_mixed=True
            )
            results_list.append(result)
        
        print(f"  MDP Policy:         {result_mdp['mean_score']:.2f} ± {result_mdp['std_score']:.2f}")
        print(f"  Adversarial Policy: {result_adv['mean_score']:.2f} ± {result_adv['std_score']:.2f}")
        print(f"  Mixed (Uniform):    {results_mixed_uniform[-1]['mean_score']:.2f} ± {results_mixed_uniform[-1]['std_score']:.2f}")
        print(f"  Mixed (Balanced):   {results_mixed_balanced[-1]['mean_score']:.2f} ± {results_mixed_balanced[-1]['std_score']:.2f}")
        print()
    
    print("SUMMARY TABLE: Mean Scores")
    print(f"\n{'Adversary':<25} {'MDP':<10} {'Adv':<10} {'Uniform':<10} {'Conserv':<10} {'Optimis':<10} {'Balanced':<10}")
    
    for i, (adv_name, _) in enumerate(adversaries):
        short_name = adv_name[:24]
        print(f"{short_name:<25} "
              f"{results_mdp[i]['mean_score']:<10.2f} "
              f"{results_adv[i]['mean_score']:<10.2f} "
              f"{results_mixed_uniform[i]['mean_score']:<10.2f} "
              f"{results_mixed_conservative[i]['mean_score']:<10.2f} "
              f"{results_mixed_optimistic[i]['mean_score']:<10.2f} "
              f"{results_mixed_balanced[i]['mean_score']:<10.2f}")
    
    all_results = {
        'MDP': results_mdp,
        'Adversarial': results_adv,
        'Mixed-Uniform': results_mixed_uniform,
        'Mixed-Conservative': results_mixed_conservative,
        'Mixed-Optimistic': results_mixed_optimistic,
        'Mixed-Balanced': results_mixed_balanced
    }


if __name__ == "__main__":
    main()