import numpy as np
import re
from collections import deque
import matplotlib.pyplot as plt
import time


class Jeu:
    def __init__(self, H):
        self.grille = np.zeros((H, 4), dtype=int)
        self.H = H #hauteur de la grille
        self.n = 0
        self.actual_piece = [] #prochaine pièce à placer
        self.coordonnees = [] #coordonnées de la prochaine pièce à placer
        self.score = 0 #score actuel
        self.state_game = True #devient False quand la partie est perdue
        self.nb_manche = 1 #nombre de manches effectuées

    def afficher_grille(self, grid):

        for i in range(self.H-1, -1, -1):
            print(grid[i])

    def strategy_bot(self, piece):

        None

    def human_play(self):
 
        # Fixed: pass the correct parameter (idx instead of grid and n separately)
        idx_current = self.state_to_idx(self.grille, self.n)
        place, term = self.placements_possibles(idx_current)
        if term: 
            print("On peut accèder à un état terminal.\n")
        for i in range(len(place)):
            print("\nUne future grille possible est :")
            self.afficher_grille(place[i][0])
            print("Avec comme reward associée :", place[i][1],"\n")

        placements = input("Entrez 3 coordonnées sous forme \"ligne, colonne\" séparées par des espaces, ou, si ce n'est pas possible, entrez \"Impossible\" : ")

        while placements != "Impossible" and not self.valid(placements):
            placements = input("Veuillez entrer une réponse valide : ")

        if placements == "Impossible":
            self.state_game = False
            return

        for elem in self.coordonnees:
            self.grille[elem[0], elem[1]] = 1

    def valid(self, placements):

        self.coordonnees = []
        placements_split = placements.split()
  
        if len(placements_split) != 3:
            return False
        
        for p in placements_split:
            if ',' not in p:
                return False
            try:
                x_string, y_string = p.split(',')
                x,y = int(x_string),int(y_string)
            except ValueError:
                return False
            if not (0 <= x <= self.H-1 and 0 <= y <= 3):
                return False
            self.coordonnees.append([x,y])
        
        if len(self.coordonnees) != len(set(tuple(c) for c in self.coordonnees)):
            print("2 coordonnées sont identiques.")
            return False

        right_place = False
        for coord in self.coordonnees:
            if self.grille[coord[0],coord[1]] == 1:
                print("Une pièce est déjà à cet endroit.")
                return False
            if coord[0]+1 <= self.H-1:
                if self.grille[coord[0]+1,coord[1]] == 1:
                    print("Vous ne pouvez pas placer une pièce sous une pièce existante.")
                    return False
            if coord[0] == 0 or self.grille[coord[0]-1, coord[1]] == 1:
                right_place = True

        if right_place == False:
            print("La pièce peut encore tomber.")
            return False

        if self.n == 1:
            if self.coordonnees[0][0] != self.coordonnees[1][0] and self.coordonnees[0][1] != self.coordonnees[1][1]:
                print("La pièce n'a pas la bonne forme.")
                return False
            if self.coordonnees[0][0] == self.coordonnees[1][0]:
                if self.coordonnees[0][0] != self.coordonnees[2][0]:
                    print("La pièce n'a pas la bonne forme.")
                    return False
            if self.coordonnees[0][1] == self.coordonnees[1][1]:
                if self.coordonnees[0][1] != self.coordonnees[2][1]:
                    print("La pièce n'a pas la bonne forme.")
                    return False

        if self.n == 2:
            distance = [(self.coordonnees[0][0]-self.coordonnees[1][0])**2 + (self.coordonnees[0][1]-self.coordonnees[1][1])**2,
                        (self.coordonnees[1][0]-self.coordonnees[2][0])**2 + (self.coordonnees[1][1]-self.coordonnees[2][1])**2,
                        (self.coordonnees[2][0]-self.coordonnees[0][0])**2 + (self.coordonnees[2][1]-self.coordonnees[0][1])**2]
            distance.sort()

            if distance[0] != distance[1] or distance[0] != 1 or distance[2] != 2:
                print("La pièce n'a pas la bonne forme.") 
                return False

        return True
    
    def valid_mute(self, placements, grid):

        coordo = []
        placements_split = placements.split()

        if len(placements_split) != 3:
            return False
        
        for p in placements_split:
            if ',' not in p:
                return False
            try:
                x_string, y_string = p.split(',')
                x,y = int(x_string),int(y_string)
            except ValueError:
                return False
            if not (0 <= x <= self.H-1 and 0 <= y <= 3):
                return False
            coordo.append([x,y])
        
        if len(coordo) != len(set(tuple(c) for c in coordo)):
            return False

        right_place = False
        for coord in coordo:
            if grid[coord[0],coord[1]] == 1:
                return False
            if coord[0]+1 <= self.H-1:
                if grid[coord[0]+1,coord[1]] == 1:
                    return False
            if coord[0] == 0 or grid[coord[0]-1, coord[1]] == 1:
                right_place = True

        if right_place == False:
            return False

        if self.n == 1:
            if coordo[0][0] != coordo[1][0] and coordo[0][1] != coordo[1][1]:
                return False
            if coordo[0][0] == coordo[1][0]:
                if coordo[0][0] != coordo[2][0]:
                    return False
            if coordo[0][1] == coordo[1][1]:
                if coordo[0][1] != coordo[2][1]:
                    return False

        if self.n == 2:
            distance = [(coordo[0][0]-coordo[1][0])**2 + (coordo[0][1]-coordo[1][1])**2,
                        (coordo[1][0]-coordo[2][0])**2 + (coordo[1][1]-coordo[2][1])**2,
                        (coordo[2][0]-coordo[0][0])**2 + (coordo[2][1]-coordo[0][1])**2]
            distance.sort()

            if distance[0] != distance[1] or distance[0] != 1 or distance[2] != 2:
                return False

        return True

    def piece(self):

        print("\n------------------------------------------------------------------------------------------\n")
        print("Manche n°:", self.nb_manche)
        self.nb_manche += 1
        print("Votre score actuel est de :", self.score)

        if self.n == 1:
            self.actual_piece = [1,1,1]
            print("Voici la pièce à placer :\n", self.actual_piece)
        
        if self.n == 2:
            self.actual_piece = [[1,0],[1,1]]
            print("Voici la pièce à placer :")
            for i in self.actual_piece:
                print(i)

        print("Voici l'état de la grille :")
        self.afficher_grille(self.grille)

    def idx_to_state(self, idx):

        p = idx % 2
        g = idx // 2
        grid = np.zeros((self.H,4), dtype=int)
        while g != 0:
            g = int(g)
            k = (g).bit_length() - 1
            g -= 2**k
            ligne = 0
            while k > 3:
                ligne += 1
                k -= 4
            grid[ligne][k] = 1
        return grid,p

    def state_to_idx(self, grid, n):

        idx = 0

        for i in range(self.H):
            for j in range(4):
                idx += int(grid[i][j])*2**(4*i+j)
        
        # Add piece type (n is 1 or 2, we store 0 or 1)
        idx = 2 * idx + (n - 1)

        return idx

    def placements_possibles(self, idx):
    
        grid, n = self.idx_to_state(idx)
        n += 1
        H = self.H
        results = []
        any_terminal = False
        seen_grids = set()

        if n == 1:
            rotations = [
                [[1,1,1]],
                [[1],[1],[1]]
            ]
        elif n == 2:
            rotations = [
                [[1,0],[1,1]],
                [[0,1],[1,1]],
                [[1,1],[0,1]],
                [[1,1],[1,0]]
            ]

        for rot in rotations:
            hauteur_piece = len(rot)
            largeur_piece = len(rot[0])

            for y in range(H - hauteur_piece + 1):
                for x in range(4 - largeur_piece + 1):

                    coords = []
                    for dy in range(hauteur_piece):
                        for dx in range(largeur_piece):
                            if rot[dy][dx] == 1:
                                coords.append([y + dy, x + dx])

                    if any(cy >= H for cy, cx in coords):
                        any_terminal = True
                        continue
                    
                    support = False
                    for cy, cx in coords:
                        if cy == 0 or grid[cy-1][cx] == 1:
                            support = True
                            break
                    if not support:
                        continue

                    placements_str = " ".join(f"{cy},{cx}" for cy,cx in coords)
                    if not self.valid_mute(placements_str, grid):
                        if any(cy >= H or grid[cy, cx] == 1 for cy, cx in coords):
                            any_terminal = True
                        continue

                    new_grid = np.copy(grid)
                    for cy, cx in coords:
                        new_grid[cy, cx] = 1
                    new_grid, reward = self.reward(new_grid)

                    grid_tuple = tuple(tuple(row) for row in new_grid)
                    if grid_tuple not in seen_grids:
                        seen_grids.add(grid_tuple)
                        results.append([new_grid, reward])
            
        return results, any_terminal
    
    def all_state(self):

        queue = deque([[np.zeros((self.H, 4), dtype=int),1],[np.zeros((self.H, 4), dtype=int),2]])
        all = []
        seen = set()
        global_to_vi = {}
        vi_to_global = {}
        cpt = 0

        while queue:
            attente = queue.popleft()
            idx_attente = self.state_to_idx(attente[0], attente[1])
            if idx_attente not in seen:

                global_to_vi[idx_attente] = cpt
                vi_to_global[cpt] = idx_attente
                cpt += 1

                next, term = self.placements_possibles(idx_attente)
                seen.add(idx_attente)
                all.append(idx_attente)
                for elem in next:
                    queue.append([elem[0],1])
                    queue.append([elem[0],2])

        return all, global_to_vi, vi_to_global

    def reward(self, grid):

        cpt = 0
        rew = 0
        erase_line = []
        for i in range(self.H):
            if sum(grid[i]) == 4:
                cpt += 1
                erase_line.append(i)
        if cpt == 1:
            rew += 1
        if cpt == 2:
            rew += 3
        if cpt == 3:
            rew += 6
        for elem in reversed(erase_line):
            grid[elem:-1] = grid[elem+1:]
            grid[-1] = 0
        return grid, rew

    def rewards_transition(self):

        reachable_global, global_to_vi, vi_to_global = self.all_state()
        R = len(reachable_global)
        actions = [[] for _ in range(R+1)]

        for s_global in reachable_global:
            vi = global_to_vi[s_global]
            successors_list, is_term = self.placements_possibles(s_global)

            if is_term:
                actions[vi].append((0, [(1, R)]))
            
            for g_next in successors_list:
                r = g_next[1]
                transitions = []

                for p_next in (1,2):
                    s_next = self.state_to_idx(g_next[0], p_next)
                    next_vi = global_to_vi[s_next]
                    transitions.append((0.5, next_vi))

                actions[vi].append((r, transitions))
                
        return actions 

    def Value_iteration(self, gamma=0.99):

        eps = 1e-9
        max_iter = 10000
        all, global_to_vi, vi_to_global = self.all_state()
        R = len(all)
        V = np.zeros(R+1)
        actions = self.rewards_transition()

        for it in range(max_iter):
            V_old = V.copy()
            policy = [-1] * R
            for i in range(R):
                best_value = 0
                best_a = None
                for action_idx, r_t in enumerate(actions[i]):
                    q = r_t[0] + gamma*sum(s[0]*V_old[s[1]] for s in r_t[1])
                    if q > best_value:
                        best_value = q
                        # Fixed: store the action index instead of trying to extract grid
                        best_a = action_idx
                V[i] = best_value
                policy[i] = best_a
            if np.max(np.abs(V - V_old)) < eps:
                print(f"Converged in {it} iterations")
                break
        return V, policy

    def Value_iteration_adversarial(self, gamma=0.99):

        eps = 1e-9
        max_iter = 10000
        all_states, global_to_vi, vi_to_global = self.all_state()
        R = len(all_states)
        V = np.zeros(R + 1)
        policy = [-1] * R

        actions = self.rewards_transition()

        for it in range(max_iter):
            V_old = V.copy()

            for i in range(R):
                best_value = 0
                best_action = None

                for action_idx, r_t in enumerate(actions[i]):
                    reward = r_t[0]
                    next_states = r_t[1]

                    if len(next_states) > 0:
                        worst_next = min(V_old[s[1]] for s in next_states)
                    else:
                        worst_next = 0

                    q = reward + gamma * worst_next

                    if q > best_value:
                        best_value = q
                        best_action = action_idx

                V[i] = best_value
                policy[i] = best_action

            if np.max(np.abs(V - V_old)) < eps: 
                print(f"Converged in {it} iterations")
                break

        return V, policy


    def place_new_piece(self):

        if self.n != 0:
            cpt = 0
            erase_line = []
            for i in range(self.H):
                if sum(self.grille[i]) == 4:
                    cpt += 1
                    erase_line.append(i)
            if cpt == 1:
                self.score += 1
            if cpt == 2:
                self.score += 3
            if cpt == 3:
                self.score += 6
            for elem in reversed(erase_line):
                self.grille[elem:-1] = self.grille[elem+1:]
                self.grille[-1] = 0

        nombre = np.random.choice([1, 2])
        self.n = nombre
        self.piece()

        #self.strategy_bot()
        self.human_play()

    def complete_game(self):

        while self.state_game != False:
            self.place_new_piece()
        print("\n------------------------------------------------------------------------------------------\n")
        print("La partie est finie.\nVous avez effectué", self.nb_manche, "manches. Le score final est de :", self.score,".\n La grille finale est :")
        self.afficher_grille(self.grille)


if __name__ == "__main__":
    H = 4
    ma_partie = Jeu(H)
    all_states, global_to_vi, vi_to_global = ma_partie.all_state()
    print(f"Number of reachable states: {len(all_states)}")
    empty_grid = np.zeros((H, 4), dtype=int)
    idx_p0 = ma_partie.state_to_idx(empty_grid, 1)
    idx_p1 = ma_partie.state_to_idx(empty_grid, 2)
    
    print("\nSTANDARD MDP")
    gamma_values = [0.90, 0.95, 0.99, 0.999]
    results_mdp = []
    
    print(f"{'Gamma':<8} {'V*(p=1)':<12} {'V*(p=2)':<12} {'Iterations':<12} {'Time(s)':<12}")
    
    for gamma in gamma_values:
        start = time.time()
        V, policy = ma_partie.Value_iteration(gamma=gamma)
        elapsed = time.time() - start
        
        vi_idx_p0 = global_to_vi[idx_p0]
        vi_idx_p1 = global_to_vi[idx_p1]
        
        print(f"{gamma:<8.3f} {V[vi_idx_p0]:<12.2f} {V[vi_idx_p1]:<12.2f} {'converged':<12} {elapsed:<12.4f}")
        
        results_mdp.append({
            'gamma': gamma,
            'V_p0': V[vi_idx_p0],
            'V_p1': V[vi_idx_p1],
            'time': elapsed
        })
    
    print("\nADVERSARIAL GAME")
    results_adv = []
    
    print(f"{'Gamma':<8} {'V*(p=1)':<12} {'V*(p=2)':<12} {'Iterations':<12} {'Time(s)':<12}")
    
    for gamma in gamma_values:
        start = time.time()
        V, policy = ma_partie.Value_iteration_adversarial(gamma=gamma)
        elapsed = time.time() - start
        
        vi_idx_p0 = global_to_vi[idx_p0]
        vi_idx_p1 = global_to_vi[idx_p1]
        
        print(f"{gamma:<8.3f} {V[vi_idx_p0]:<12.2f} {V[vi_idx_p1]:<12.2f} {'converged':<12} {elapsed:<12.4f}")
        
        results_adv.append({
            'gamma': gamma,
            'V_p0': V[vi_idx_p0],
            'V_p1': V[vi_idx_p1],
            'time': elapsed
        })