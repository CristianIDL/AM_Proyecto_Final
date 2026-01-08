import numpy as np
import pickle
import time
import os
import glob
import sys

# ==========================================
# CONFIGURACIÓN (Debe coincidir con el entrenamiento)
# ==========================================

class GameConfig:
    MAP_SIZE = 18  
    
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
            int(40 * 2/3), 40, int(160 * 1/3), int(160 * 2/3), 160,
            int(640 * 1/3), int(640 * 2/3), 640,
            int(2560 * 1/3), int(2560 * 2/3), 2560, 999999
        ]
        return reqs[level_idx] if level_idx < len(reqs) else 999999

# ENTORNO DE JUEGO (Solo lectura)

class MapEnvironment:
    def __init__(self, maps_folder='mapas'):
        self.maps_folder = maps_folder
        self.map_templates = self._load_maps()
        self.maps = []
        self.current_map_idx = 0
        self.agent_pos = (1, 1)
        self.max_steps = 300
        self.steps = 0

    def _load_maps(self):
        if not os.path.exists(self.maps_folder):
            raise ValueError(f" Carpeta '{self.maps_folder}' no encontrada")

        files = sorted(glob.glob(os.path.join(self.maps_folder, "*.txt")))
        loaded_maps = []
        
        print(f"Cargando mapas desde: {self.maps_folder}")
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_lines = f.readlines()
            
            # Limpiar líneas
            lines = []
            for line in raw_lines:
                cleaned = line.replace('\n', '').replace('\r', '')
                if cleaned:
                    lines.append(cleaned)
            
            # 🔧 FIX: Ajustar a 18x18
            if len(lines) > 18:
                lines = lines[:18]
            elif len(lines) < 18:
                lines += ['#' * 18] * (18 - len(lines))
            
            grid_list = []
            for line in lines:
                if len(line) < 18:
                    l = list(line.ljust(18, '#'))
                else:
                    l = list(line[:18])
                grid_list.append(l)
                
            loaded_maps.append(np.array(grid_list))
            print(f"   ✓ {os.path.basename(file_path)} ({len(grid_list)}x{len(grid_list[0])})")
            
        if not loaded_maps:
            raise ValueError("No se encontraron mapas válidos")
            
        return loaded_maps

    def reset(self, keep_progress=False):
        if not keep_progress:
            self.current_map_idx = 0
            self.maps = [m.copy() for m in self.map_templates]
        
        if self.current_map_idx >= len(self.maps):
            self.current_map_idx = 0
            self.maps = [m.copy() for m in self.map_templates]

        current_map = self.maps[self.current_map_idx]
        start_positions = np.where(current_map == 'S')
        
        if len(start_positions[0]) > 0:
            self.agent_pos = (start_positions[0][0], start_positions[1][0])
        else:
            empty_positions = np.where((current_map == ' ') | (current_map == "'"))
            if len(empty_positions[0]) > 0:
                self.agent_pos = (empty_positions[0][0], empty_positions[1][0])
            else:
                self.agent_pos = (1, 1)
            
        self.steps = 0
        return self.get_vision()

    def get_vision(self):
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
        
        if not (0 <= ny < 18 and 0 <= nx < 18):
            ny, nx = y, x

        cell = current_map[ny, nx]
        reward = -1
        done = False
        info = {}

        if cell == GameConfig.WALL:
            reward = -5
            ny, nx = y, x
            
        elif cell == GameConfig.TREASURE:
            bonus = 10 + (agent_rod_lvl * 5)
            reward = bonus
            info['gold_gain'] = bonus
            current_map[ny, nx] = "'"  # Usa ' si tu mapa usa ', o ' ' si usa espacios
            
        elif cell in [GameConfig.ENEMY_X, GameConfig.ENEMY_E]:
            reward = -50
            
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
                ny, nx = y, x  # No puede pasar

        # 🔧 FIX: Actualizar posición DESPUÉS de todas las validaciones
        self.agent_pos = (ny, nx)
        
        if self.steps >= self.max_steps:
            done = True
            info['timeout'] = True
            
        return self.get_vision(), reward, done, info

# CODIFICADOR DE ESTADO 

class StateEncoder:
    """Convierte visión 5x5 + atributos + posición en clave única"""
    
    def encode(self, vision, gold, map_idx, rod_lvl, agent_pos):
        req = GameConfig.get_gold_requirement(map_idx)
        can_exit = gold >= req
        
        # --- CORRECCIÓN AQUÍ ---
        # Antes solo mirabas si era pared o no.
        # Ahora distinguimos: 0=Vacío, 1=Pared, 2=Tesoro, 3=Salida
        surroundings = []
        cx, cy = 2, 2 # Centro de la visión
        
        # Norte, Sur, Este, Oeste
        for dy, dx in [(0,-1), (0,1), (1,0), (-1,0)]:
            cell = vision[cy+dy, cx+dx]
            
            if cell == '#': 
                val = 1 # Pared
            elif cell == 'T': 
                val = 2 # Tesoro (¡IMPORTANTE!)
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

# ==========================================
# AGENTE (Solo inferencia)
# ==========================================

class Agent:
    def __init__(self, pkl_path):
        self.encoder = StateEncoder()
        self.load_agent(pkl_path)
        self.gold = 0
        self.rod_lvl = 1

    def load_agent(self, filepath):
        print(f"\nCargando agente desde: {filepath}")
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        self.q_tables = agent_data['q_tables']
        self.epsilon = 0.0  # Sin exploración
        
        total_states = sum(len(table) for table in self.q_tables)
        print(f"    {len(self.q_tables)} mapas cargados")
        print(f"   {total_states} estados aprendidos")
        print(f"   Entrenado el: {agent_data.get('timestamp', 'desconocido')}")

    def get_action(self, vision, map_idx, agent_pos):
        state = self.encoder.encode(vision, self.gold, map_idx, self.rod_lvl, agent_pos)
        
        if map_idx >= len(self.q_tables):
            return np.random.randint(4)  # Fallback
        
        table = self.q_tables[map_idx]
        
        if state not in table:
            # Estado nunca visto: exploración inteligente
            return np.random.randint(4)
        
        # Política greedy pura
        max_val = np.max(table[state])
        best_actions = [i for i, v in enumerate(table[state]) if v == max_val]
        return np.random.choice(best_actions)

# VISUALIZADOR MEJORADO

def render_game(env, agent, stats):
    """Renderiza el estado del juego con interfaz mejorada"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Header
    print(f"{' VISUALIZADOR DE AGENTE ENTRENADO':^70}")
    
    # Stats superior
    req = GameConfig.get_gold_requirement(env.current_map_idx)
    print(f"\n MAPA: {env.current_map_idx + 1}/{len(env.maps)}")
    print(f" ORO: {agent.gold:4d} / {req:4d} {' ' if agent.gold >= req else 'no'}")
    print(f" ROD: Nivel {agent.rod_lvl} (Bonus: +{agent.rod_lvl * 5} oro)")
    print(f" PASOS: {env.steps}/{env.max_steps}")
    print(f" REWARD TOTAL: {stats['total_reward']:.0f}")
    print(f"  MAPAS COMPLETADOS: {stats['maps_cleared']}")
    
    print("\n" + "-" * 70)
    
    m = env.map_templates[env.current_map_idx].copy()
    
    # Superponer cambios del mapa actual (tesoros consumidos)
    current_map = env.maps[env.current_map_idx]
    for i in range(18): 
        for j in range(18):  
            # Si en el mapa actual hay espacio/comilla pero en template había T, se consumió
            if current_map[i, j] in [' ', "'"] and m[i, j] == 'T':
                m[i, j] = current_map[i, j]  # Mostrar como consumido
    
    # Mostrar agente
    ay, ax = env.agent_pos
    original_cell = m[ay, ax]
    
    if original_cell in ['R', 'G']:
        m[ay, ax] = '◉'  # Círculo para indicar que está sobre la salida
    else:
        m[ay, ax] = 'A'
    
    # Leyenda lateral
    legend = [
        "LEYENDA:",
        "A = Agente",
        "◉ = Agente en salida",
        "S = Inicio",
        "R/G = Salida",
        "T = Tesoro",
        "P = Tienda",
        "X/E = Enemigo"
    ]
    
    print("    " + "".join([f"{i:2d}" for i in range(18)]))  # 🔧 FIX: 18 columnas
    for i, row in enumerate(m):
        line = f"{i:2d}  " + "".join(row)
        if i < len(legend):
            line += f"    {legend[i]}"
        print(line)
    
    print("-" * 70)

def main():
    print("\n" + "=" * 70)
    print(f"{'🎮 VISUALIZADOR DE AGENTE Q-LEARNING':^70}")
    print("=" * 70 + "\n")
    
    # CONFIGURACIÓN
    
    # Solicitar rutas
    maps_folder = input(" Carpeta de mapas (default: 'mapas'): ").strip() or 'mapas'
    pkl_path = input(" Archivo .pkl del agente (default: 'agente_entrenado.pkl'): ").strip() or 'agente_entrenado.pkl'
    
    # Velocidad de reproducción
    print("\n⏱  Velocidad de reproducción:")
    print("   1 = Muy rápida (0.05s)")
    print("   2 = Rápida (0.1s)")
    print("   3 = Normal (0.2s)")
    print("   4 = Lenta (0.5s)")
    speed_choice = input("Selecciona (1-4, default: 2): ").strip() or '2'
    
    speeds = {'1': 0.05, '2': 0.1, '3': 0.2, '4': 0.5}
    delay = speeds.get(speed_choice, 0.1)
    
    # INICIALIZACIÓN
    
    try:
        env = MapEnvironment(maps_folder=maps_folder)
        agent = Agent(pkl_path=pkl_path)
    except Exception as e:
        print(f"\n Error al cargar: {e}")
        sys.exit(1)
    
    print(f"\n Todo listo. Iniciando visualización...\n")
    time.sleep(2)
    
    # LOOP DE VISUALIZACIÓN
    
    vision = env.reset(keep_progress=False)
    agent.gold = 0
    agent.rod_lvl = 1
    
    stats = {
        'total_reward': 0,
        'maps_cleared': 0,
        'total_steps': 0,
        'treasures_collected': 0,
        'upgrades_bought': 0
    }
    
    max_total_steps = 10000
    
    try:
        while stats['total_steps'] < max_total_steps:
            # Renderizar
            render_game(env, agent, stats)
            
            # Decisión del agente
            action = agent.get_action(vision, env.current_map_idx, env.agent_pos)
            
            # Ejecutar acción
            old_pos = env.agent_pos
            vision, reward, done, info = env.step(action, agent.gold, agent.rod_lvl)
            
            # Actualizar stats del agente
            if 'gold_gain' in info:
                agent.gold += info['gold_gain']
                stats['treasures_collected'] += 1
                
            if 'upgrade' in info:
                agent.gold -= info['cost']
                agent.rod_lvl += 1
                stats['upgrades_bought'] += 1
            
            stats['total_reward'] += reward
            stats['total_steps'] += 1
            
            time.sleep(delay)
            
            # Manejar fin de nivel/juego
            if done:
                render_game(env, agent, stats)
                
                if info.get('level_complete'):
                    print(f"{'¡NIVEL COMPLETADO!':^70}")
                    
                    stats['maps_cleared'] += 1
                    time.sleep(2)
                    
                    env.current_map_idx += 1
                    if env.current_map_idx >= len(env.maps):
                        print("\n" + "" * 35)
                        print(f"{'¡¡¡ VICTORIA TOTAL !!!':^70}")
                        print("" * 35)
                        print(f"\n ESTADÍSTICAS FINALES:")
                        print(f"   • Mapas completados: {stats['maps_cleared']}")
                        print(f"   • Tesoros recolectados: {stats['treasures_collected']}")
                        print(f"   • Mejoras compradas: {stats['upgrades_bought']}")
                        print(f"   • Pasos totales: {stats['total_steps']}")
                        print(f"   • Reward total: {stats['total_reward']:.0f}")
                        break
                    
                    vision = env.reset(keep_progress=True)
                    
                elif info.get('timeout'):
                    print("\n" + " " * 35)
                    print(f"{'TIMEOUT - Se acabó el tiempo':^70}")
                    print(" " * 35)
                    time.sleep(2)
                    break
                    
                else:
                    print("\n" + "" * 35)
                    print(f"{'GAME OVER':^70}")
                    print("" * 35)
                    print(f"\n ESTADÍSTICAS:")
                    print(f"   • Mapas completados: {stats['maps_cleared']}")
                    print(f"   • Tesoros recolectados: {stats['treasures_collected']}")
                    print(f"   • Pasos totales: {stats['total_steps']}")
                    print(f"   • Reward total: {stats['total_reward']:.0f}")
                    break
    
    except KeyboardInterrupt:
        print("\n\n Visualización interrumpida por usuario")
        print(f"\n ESTADÍSTICAS PARCIALES:")
        print(f"   • Mapas completados: {stats['maps_cleared']}")
        print(f"   • Pasos totales: {stats['total_steps']}")
    
    print(f"{' Visualización finalizada':^70}")

# PUNTO DE ENTRADA

if __name__ == "__main__":
    main()