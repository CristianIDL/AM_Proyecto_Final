import numpy as np
import random
import time
import os
import glob
import pickle
import json
from datetime import datetime

# 1. CONFIGURACIÓN Y REGLAS

class GameConfig:
    MAP_SIZE = 18  
    
    # Símbolos del Mapa
    WALL = '#'
    EMPTY = ' '
    START = 'S'
    EXIT_R = 'R'
    EXIT_G = 'G'
    TREASURE = 'T'
    ENEMY_X = 'X'
    ENEMY_E = 'E'
    SHOP = 'P'

    @staticmethod
    def get_gold_requirement(level_idx):
        reqs = [
            int(40 * 2/3),   # Mapa 0: ~26
            40,              # Mapa 1: 40
            int(160 * 1/3),  # Mapa 2: ~53
            int(160 * 2/3),  # Mapa 3: ~106
            160,             # Mapa 4
            int(640 * 1/3),  # Mapa 5: ~213
            int(640 * 2/3),  # Mapa 6: ~426
            640,             # Mapa 7
            int(2560 * 1/3), # Mapa 8
            int(2560 * 2/3), # Mapa 9
            2560,            # Mapa 10
            999999           # Mapa 11+
        ]
        if level_idx < len(reqs):
            return reqs[level_idx]
        return 999999

# 2. ENTORNO DE JUEGO

class MapEnvironment:
    def __init__(self, maps_folder='mapas'):
        self.maps_folder = maps_folder
        self.map_templates = self._load_maps()
        self.maps = []
        
        self.current_map_idx = 0
        self.agent_pos = (1, 1)
        self.max_steps = 500
        self.steps = 0

    def _load_maps(self):
        if not os.path.exists(self.maps_folder):
            os.makedirs(self.maps_folder)

        files = sorted(glob.glob(os.path.join(self.maps_folder, "*.txt")))
        loaded_maps = []
        
        print(f"Cargando mapas desde {self.maps_folder}...")
        
        # USAR EL TAMAÑO DE LA CONFIGURACIÓN
        SIZE = GameConfig.MAP_SIZE 
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.replace('\n', '').replace('\r', '') for line in f.readlines() if line.strip()]
            
            # 1. Ajustar ALTO (Filas)
            if len(lines) < SIZE:
                lines += ['#' * SIZE] * (SIZE - len(lines))
            elif len(lines) > SIZE:
                lines = lines[:SIZE] # Recortar si sobra
            
            grid_list = []
            for line in lines:
                # 2. Ajustar ANCHO (Columnas) con GameConfig.MAP_SIZE
                # ljust rellena con '#', slicing [:SIZE] corta lo que sobra
                l = list(line.ljust(SIZE, '#'))[:SIZE] 
                grid_list.append(l)
                
            loaded_maps.append(np.array(grid_list))
            print(f" -> {os.path.basename(file_path)} cargado ({SIZE}x{SIZE}).")
            
        if not loaded_maps:
            raise ValueError("No se encontraron mapas válidos.")
            
        return loaded_maps

    def reset(self, keep_progress=False):
        """
        keep_progress=True: Avanza de nivel
        keep_progress=False: Reinicia desde mapa 0
        """
        if not keep_progress:
            self.current_map_idx = 0
            self.maps = [m.copy() for m in self.map_templates]
        
        if self.current_map_idx >= len(self.maps):
            self.current_map_idx = 0
            self.maps = [m.copy() for m in self.map_templates]

        # BUSCAR POSICIÓN 'S' CORRECTAMENTE
        current_map = self.maps[self.current_map_idx]
        start_positions = np.where(current_map == 'S')
        
        if len(start_positions[0]) > 0:
            self.agent_pos = (start_positions[0][0], start_positions[1][0])
            
        self.steps = 0
        return self.get_vision()

    def get_vision(self):
        """Extrae ventana 5x5"""
        y, x = self.agent_pos
        current_map = self.maps[self.current_map_idx]
        
        padded = np.pad(current_map, 2, constant_values='#')
        py, px = y + 2, x + 2
        
        return padded[py-2:py+3, px-2:px+3]

    def step(self, action, agent_gold, agent_rod_lvl):
        self.steps += 1
        y, x = self.agent_pos
        
        moves = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        dy, dx = moves[action]
        ny, nx = y + dy, x + dx
        
        current_map = self.maps[self.current_map_idx]
        
        # CORRECCIÓN DE LÍMITES: Usar GameConfig.MAP_SIZE
        limit = GameConfig.MAP_SIZE
        if not (0 <= ny < limit and 0 <= nx < limit):
            ny, nx = y, x # Choque con límite del mapa

        cell = current_map[ny, nx]
        
        reward = -1
        done = False
        info = {}

        if cell == GameConfig.WALL:
            reward = -5
            ny, nx = y, x
            
        elif cell == GameConfig.TREASURE:
            bonus = 10 + (agent_rod_lvl * 5) # Ejemplo de bonus
            reward = bonus
            info['gold_gain'] = bonus
            current_map[ny, nx] = "'" 
            
        elif cell == GameConfig.ENEMY_X:
            reward = -500 
            # done = True # Opcional: morir
        elif cell == GameConfig.ENEMY_E: 
            reward = -300 
            # PONER MODULO DE COMBATE AQUÍ
            
        elif cell == GameConfig.SHOP:
            cost = agent_rod_lvl * 50
            if agent_gold >= cost:
                reward = 50 
                info['upgrade'] = True
                info['cost'] = cost
            else:
                reward = -2
                
        elif cell in [GameConfig.EXIT_R, GameConfig.EXIT_G]:
            req = GameConfig.get_gold_requirement(self.current_map_idx)
            if agent_gold >= req:
                reward = 200
                done = True
                info['level_complete'] = True
            else:
                reward = -20
                ny, nx = y, x

        self.agent_pos = (ny, nx)
        
        if self.steps >= self.max_steps:
            done = True
            
        return self.get_vision(), reward, done, info

# 3. CODIFICADOR DE ESTADO MEJORADO

class StateEncoder:
    """Convierte visión 5x5 + atributos + posición en clave única"""
    
    def encode(self, vision, gold, map_idx, rod_lvl, agent_pos):
        req = GameConfig.get_gold_requirement(map_idx)
        can_exit = gold >= req
        
        #  0=Vacío, 1=Pared, 2=Tesoro, 3=Salida
        surroundings = []
        cx, cy = 2, 2 # Centro de la visión
        
        # Norte, Sur, Este, Oeste
        for dy, dx in [(0,-1), (0,1), (1,0), (-1,0)]:
            cell = vision[cy+dy, cx+dx]
            
            if cell == '#': 
                val = 1 # Pared
            elif cell == 'T': 
                val = 2 # Tesoro 
            elif cell in ['R', 'G']: 
                val = 3 # Salida
            elif cell in ['X', 'E']:
                val = 4 # Peligro
            else: 
                val = 0 # Espacio vacío / Camino seguro
                
            surroundings.append(val)
            
        surroundings = tuple(surroundings)
        # -----------------------
        
        pos_exact = (agent_pos[0], agent_pos[1])
        
        # Simplificamos el nivel de caña para reducir estados
        rod_tier = 0 if rod_lvl < 3 else 1 
        
        # Detectar qué hay bajo los pies
        center_cell = vision[2, 2]
        on_exit = 1 if center_cell in ['R', 'G'] else 0
        
        # Retornamos 'surroundings' en lugar de 'walls'
        return (can_exit, surroundings, pos_exact, rod_tier, on_exit)

# 4. AGENTE CON EXPORTACIÓN

class PersistentAgent:
    def __init__(self, num_maps=12):
        self.q_tables = [{} for _ in range(num_maps)]
        self.encoder = StateEncoder()
        
        self.gold = 0
        self.rod_lvl = 1
        
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
        self.last_positions = []
        
        # Estadísticas de entrenamiento
        self.training_stats = {
            'episodes_completed': 0,
            'total_rewards': [],
            'maps_cleared_per_episode': [],
            'epsilon_history': []
        }

    def get_action(self, vision, map_idx, agent_pos):
        state = self.encoder.encode(vision, self.gold, map_idx, self.rod_lvl, agent_pos)
        table = self.q_tables[map_idx]
        
        if state not in table:
            table[state] = np.zeros(4)
        
        # Detector de ciclos
        if len(self.last_positions) >= 4:
            recent = self.last_positions[-4:]
            if len(set(recent)) <= 2:
                return np.random.randint(4)
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        
        max_val = np.max(table[state])
        best_actions = [i for i, v in enumerate(table[state]) if v == max_val]
        return np.random.choice(best_actions)

    def learn(self, vision, action, reward, next_vision, map_idx, done, agent_pos, next_agent_pos):
        state = self.encoder.encode(vision, self.gold, map_idx, self.rod_lvl, agent_pos)
        next_state = self.encoder.encode(next_vision, self.gold, map_idx, self.rod_lvl, next_agent_pos)
        
        table = self.q_tables[map_idx]
        if state not in table: 
            table[state] = np.zeros(4)
        if next_state not in table: 
            table[next_state] = np.zeros(4)
        
        target = reward
        if not done:
            target += self.gamma * np.max(table[next_state])
            
        predict = table[state][action]
        table[state][action] += self.lr * (target - predict)
        
        self.last_positions.append(agent_pos)
        if len(self.last_positions) > 10:
            self.last_positions.pop(0)

    #  FUNCIONES DE EXPORTACIÓN
    
    def save_agent(self, filepath='agente_entrenado.pkl'):
        """Guarda el agente completo (Q-tables + parámetros)"""
        agent_data = {
            'q_tables': self.q_tables,
            'epsilon': self.epsilon,
            'lr': self.lr,
            'gamma': self.gamma,
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f" Agente guardado en: {filepath}")
        print(f"   - Q-Tables: {len(self.q_tables)} mapas")
        print(f"   - Estados totales: {sum(len(table) for table in self.q_tables)}")
    
    def load_agent(self, filepath='agente_entrenado.pkl'):
        """Carga un agente previamente entrenado"""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        self.q_tables = agent_data['q_tables']
        self.epsilon = agent_data['epsilon']
        self.lr = agent_data['lr']
        self.gamma = agent_data['gamma']
        self.training_stats = agent_data.get('training_stats', {})
        
        print(f"Agente cargado desde: {filepath}")
        print(f"   - Entrenado el: {agent_data.get('timestamp', 'desconocido')}")
        print(f"   - Estados aprendidos: {sum(len(table) for table in self.q_tables)}")
    
    def export_qtable_readable(self, map_idx=0, filepath='qtable_mapa_0.txt', top_n=50):
        """Exporta Q-table legible en texto (top N estados)"""
        if map_idx >= len(self.q_tables):
            print(f" Mapa {map_idx} no existe")
            return
        
        table = self.q_tables[map_idx]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Q-TABLE PARA MAPA {map_idx}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total de estados: {len(table)}\n")
            f.write(f"Acciones: 0=Norte, 1=Sur, 2=Este, 3=Oeste\n\n")
            
            # Ordenar por valor máximo de Q
            sorted_states = sorted(table.items(), 
                                   key=lambda x: np.max(x[1]), 
                                   reverse=True)
            
            f.write(f"TOP {min(top_n, len(sorted_states))} ESTADOS MÁS VALIOSOS:\n")
            f.write("-" * 80 + "\n\n")
            
            for i, (state, q_values) in enumerate(sorted_states[:top_n]):
                f.write(f"Estado #{i+1}:\n")
                f.write(f"  Descripción: {state}\n")
                f.write(f"  Q-Valores: N={q_values[0]:.2f}, S={q_values[1]:.2f}, "
                       f"E={q_values[2]:.2f}, O={q_values[3]:.2f}\n")
                f.write(f"  Mejor acción: {['Norte', 'Sur', 'Este', 'Oeste'][np.argmax(q_values)]}\n")
                f.write("\n")
        
        print(f" Q-Table exportada a: {filepath}")
    
    def export_qtable_json(self, filepath='qtables_completas.json'):
        """Exporta todas las Q-tables en formato JSON"""
        # Convertir q_tables a formato serializable
        serializable_tables = []
        for table in self.q_tables:
            serializable_table = {}
            for state, q_values in table.items():
                # Convertir tupla a string para usar como key en JSON
                state_key = str(state)
                serializable_table[state_key] = q_values.tolist()
            serializable_tables.append(serializable_table)
        
        data = {
            'q_tables': serializable_tables,
            'metadata': {
                'num_maps': len(self.q_tables),
                'total_states': sum(len(table) for table in self.q_tables),
                'epsilon': self.epsilon,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Q-Tables JSON exportadas a: {filepath}")

# 5. LOOP PRINCIPAL

if __name__ == "__main__":
    env = MapEnvironment()
    agent = PersistentAgent(num_maps=15)
    
    EPISODES = 7500
    print(f"   INICIANDO ENTRENAMIENTO ({EPISODES} Episodios)")
    
    try:
        for episode in range(EPISODES):
            vision = env.reset(keep_progress=False)
            agent.gold = 0
            agent.rod_lvl = 1
            agent.last_positions = []  # Reset historial de posiciones
            
            total_reward = 0
            maps_cleared = 0
            
            while True:
                current_map = env.current_map_idx
                old_pos = env.agent_pos
                
                # 1. Decidir
                action = agent.get_action(vision, current_map, env.agent_pos)
                
                # 2. Actuar
                next_vision, reward, done, info = env.step(action, agent.gold, agent.rod_lvl)
                
                # 3. Actualizar Stats RPG
                if 'gold_gain' in info: 
                    agent.gold += info['gold_gain']
                if 'upgrade' in info: 
                    agent.gold -= info['cost']
                    agent.rod_lvl += 1
                
                # 4. Aprender
                level_finished = info.get('level_complete', False)
                agent.learn(vision, action, reward, next_vision, current_map, done, 
                          old_pos, env.agent_pos)
                
                vision = next_vision
                total_reward += reward
                
                if done:
                    if level_finished:
                        maps_cleared += 1
                        env.current_map_idx += 1
                        if env.current_map_idx >= len(env.maps):
                            #print(f" ¡JUEGO COMPLETADO en Episodio {episode}!")
                            break
                        vision = env.reset(keep_progress=True)
                    else:
                        break
            
            # Actualizar estadísticas
            agent.training_stats['total_rewards'].append(total_reward)
            agent.training_stats['maps_cleared_per_episode'].append(maps_cleared)
            agent.training_stats['epsilon_history'].append(agent.epsilon)
            agent.training_stats['episodes_completed'] = episode + 1
            
            agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
            
            if episode % 50 == 0:
                avg_reward = np.mean(agent.training_stats['total_rewards'][-50:]) if len(agent.training_stats['total_rewards']) >= 50 else total_reward
                print(f"Ep: {episode:4d} | ε: {agent.epsilon:.3f} | Mapas: {maps_cleared} | "
                      f"Oro: {agent.gold:4d} | Reward: {total_reward:6.0f} | Avg50: {avg_reward:6.0f}")

    except KeyboardInterrupt:
        print("\n  Entrenamiento interrumpido por usuario.")
    
    #  EXPORTAR AGENTE Y Q-TABLES
    print("\n" + "="*60)
    print("   GUARDANDO RESULTADOS DEL ENTRENAMIENTO")
    print("="*60 + "\n")
    
    # Guardar agente completo
    agent.save_agent('agente_entrenado.pkl')
    
    # Exportar Q-tables legibles
    agent.export_qtable_readable(map_idx=0, filepath='qtable_mapa_0.txt', top_n=100)
    
    # Exportar todas las Q-tables en JSON
    agent.export_qtable_json('qtables_completas.json')
    
    print("\nEntrenamiento completado y guardado.\n")
    
    # MODO DEMOSTRACIÓN
    print("\n" + "="*60)
    print("   MODO DEMOSTRACIÓN")
    print("="*60)
    print("Presiona ENTER para ver al agente entrenado en acción...")
    input()
    
    agent.epsilon = 0  # Sin exploración
    vision = env.reset(keep_progress=False)
    agent.gold = 0
    agent.rod_lvl = 1
    
    demo_steps = 0
    max_demo_steps = 5000
    
    while demo_steps < max_demo_steps:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        req = GameConfig.get_gold_requirement(env.current_map_idx)
        print(f"MAPA: {env.current_map_idx} | ORO: {agent.gold}/{req} | ROD: Lv.{agent.rod_lvl} | PASO: {demo_steps}")
        print("-" * 40)
        
        # Renderizar mapa
        m = env.maps[env.current_map_idx].copy()
        ay, ax = env.agent_pos
        m[ay, ax] = 'A'
        for row in m:
            print("".join(row))
        
        action = agent.get_action(vision, env.current_map_idx, env.agent_pos)
        old_pos = env.agent_pos
        vision, _, done, info = env.step(action, agent.gold, agent.rod_lvl)
        
        if 'gold_gain' in info: 
            agent.gold += info['gold_gain']
        if 'upgrade' in info: 
            agent.gold -= info['cost']
            agent.rod_lvl += 1
        
        time.sleep(0.1)
        demo_steps += 1
        
        if done:
            if info.get('level_complete'):
                print("\n>>>  NIVEL COMPLETADO <<<")
                time.sleep(1.5)
                env.current_map_idx += 1
                if env.current_map_idx >= len(env.maps):
                    print("\n¡VICTORIA TOTAL!")
                    break
                vision = env.reset(keep_progress=True)
            else:
                print("\n>>>  GAME OVER <<<")
                break
    
    print("\n Demostración finalizada.")