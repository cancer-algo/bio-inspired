import pygame
import numpy as np
from sklearn.decomposition import PCA

class FeatureSelectionGameGUI:
    def __init__(self, algorithm, feature_names):
        pygame.init()
        self.algorithm = algorithm
        self.feature_names = feature_names
        self.num_features = len(feature_names)

        self.screen_width = 1000
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Bio-Inspired Feature Selection")

        self.font = pygame.font.SysFont('arial', 18)
        self.cell_size = 100
        self.grid_offset_x = 20
        self.grid_offset_y = 120

        usable_width = self.screen_width - 2 * self.grid_offset_x
        usable_height = self.screen_height - self.grid_offset_y - 20
        self.grid_cols = max(1, usable_width // self.cell_size)
        self.grid_rows = max(1, -(-self.num_features // self.grid_cols))


        self.clock = pygame.time.Clock()
        self.paused = False
        self.current_iteration = 0
        self.history = []
        self.log = []

        self.pca = PCA(n_components=2)
        self.agent_positions = [{'current': None, 'target': None, 't': 1.0}
                                for _ in range(algorithm.num_agents)]
        self.best_position = None

    def draw_grid(self, feature_mask):
        for i in range(self.num_features):
            row = i // self.grid_cols
            col = i % self.grid_cols
            x = self.grid_offset_x + col * self.cell_size
            y = self.grid_offset_y + row * self.cell_size
            color = (0, 255, 0) if feature_mask[i] else (80, 80, 80)
            pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size), 2)

            name = self.feature_names[i][:10]
            text = self.font.render(name, True, (255, 255, 255))
            self.screen.blit(text, (x + 5, y + self.cell_size // 2 - 10))

    def update_agent_positions(self):
        try:
            all_positions = np.vstack([self.algorithm.population, self.algorithm.global_best])
            coords = self.pca.fit_transform(all_positions)

            min_vals = coords.min(axis=0)
            max_vals = coords.max(axis=0)
            coords = (coords - min_vals) / (max_vals - min_vals + 1e-6)

            for i, agent in enumerate(self.agent_positions):
                if agent['target'] is not None:
                    agent['current'] = agent['target']
                else:
                    agent['current'] = coords[i]
                agent['target'] = coords[i]
                agent['t'] = 0.0

            self.best_position = coords[-1]

        except Exception as e:
            print(f"[PCA Error] {e}")
            for agent in self.agent_positions:
                agent['current'] = agent.get('target', np.random.rand(2))
                agent['target'] = np.random.rand(2)
                agent['t'] = 0.0
            self.best_position = np.random.rand(2)

    def draw_agents(self):
        for agent in self.agent_positions:
            if agent['current'] is not None and agent['target'] is not None:
                agent['t'] = min(agent['t'] + 0.05, 1.0)
                pos = agent['current'] + (agent['target'] - agent['current']) * agent['t']
                x = int(self.grid_offset_x + pos[0] * self.grid_cols * self.cell_size)
                y = int(self.grid_offset_y + pos[1] * self.grid_rows * self.cell_size)
                pygame.draw.circle(self.screen, (0, 150, 255), (x, y), 8)

        if self.best_position is not None:
            x = int(self.grid_offset_x + self.best_position[0] * self.grid_cols * self.cell_size)
            y = int(self.grid_offset_y + self.best_position[1] * self.grid_rows * self.cell_size)
            pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 12)
            text = self.font.render("Best", True, (255, 255, 255))
            self.screen.blit(text, (x + 10, y - 10))

    def draw_scoreboard(self, iteration, fitness, selected_features):
        self.screen.fill((0, 0, 0), (0, 0, self.screen_width, 100))
        lines = [
            f"Iteration: {iteration}",
            f"Fitness: {fitness:.5f}",
            f"Selected: {', '.join(selected_features)[:60]}",
            f"Paused: {self.paused}",
        ]
        for i, line in enumerate(lines):
            surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(surface, (10, 10 + i * 20))

        log_y = 10
        for i in range(max(0, len(self.log) - 3), len(self.log)):
            log_surface = self.font.render(self.log[i], True, (200, 200, 200))
            self.screen.blit(log_surface, (700, log_y))
            log_y += 20

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True

    def run(self):
        running = True
        self.exit_gui = False
        while running:
            running = self.handle_events()
            self.screen.fill((30, 30, 30))

            if not self.paused and self.algorithm.cur_iter < self.algorithm.max_iter:
                self.algorithm.next()
                self.update_agent_positions()

                best_vec = np.array(self.algorithm.global_best)
                fitness = self.algorithm.global_best_fitness
                selected = [self.feature_names[i] for i, b in enumerate(best_vec) if b]

                self.current_iteration = self.algorithm.cur_iter
                self.history.append((best_vec, fitness))
            else:
                best_vec = np.array(self.algorithm.global_best)
                fitness = self.algorithm.global_best_fitness
                selected = [self.feature_names[i] for i, b in enumerate(best_vec) if b]

            self.draw_grid(best_vec)
            self.draw_agents()
            self.draw_scoreboard(self.current_iteration, fitness, selected)
            pygame.display.flip()
            self.clock.tick(10)

        self.exit_gui = True
        with open(f"output_{self.algorithm.algo_name.lower()}.txt", 'a') as f:
            for log_entry in self.log:
                f.write(log_entry + "\n")
        pygame.quit()
