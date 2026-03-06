# space_invaders_astar_autopilot.py
import os
import random
import time
import math
import heapq
import pygame as pg

# -------------------------
# CONFIG
# -------------------------
WIDTH, HEIGHT = 1000, 800
FPS = 60

# Player
PLAYER_W, PLAYER_H = 64, 64
PLAYER_VEL = 8

# Bullet
BULLET_W, BULLET_H = 8, 20
BULLET_VEL = -10
SHOOT_COOLDOWN = 300  # ms

# Enemy (rocks)
ENEMY_W, ENEMY_H = 48, 48
ENEMY_VEL = 3
ENEMY_SPAWN_INTERVAL = 1200  # ms
ENEMY_ANIM_ROT_SPEED = 2  # degrees per frame

ASSET_DIR = "."

# -------------------------
# Utilities
# -------------------------
def safe_image_load(names, size=None):
    for name in names:
        path = os.path.join(ASSET_DIR, name)
        if os.path.exists(path):
            try:
                img = pg.image.load(path).convert_alpha()
                if size:
                    img = pg.transform.smoothscale(img, size)
                return img
            except Exception:
                pass
    return None

# -------------------------
# Game objects
# -------------------------
class Bullet:
    def __init__(self, x, y, img=None):
        self.rect = pg.Rect(x, y, BULLET_W, BULLET_H)
        self.img = img

    def update(self):
        self.rect.y += BULLET_VEL

    def draw(self, surface):
        if self.img:
            surface.blit(self.img, (self.rect.x, self.rect.y))
        else:
            pg.draw.rect(surface, (255, 255, 0), self.rect)

class Enemy:
    def __init__(self, x, y, img=None):
        self.rect = pg.Rect(x, y, ENEMY_W, ENEMY_H)
        self.img = img
        self.angle = 0

    def update(self):
        self.rect.y += ENEMY_VEL
        self.angle = (self.angle + ENEMY_ANIM_ROT_SPEED) % 360

    def draw(self, surface):
        if self.img:
            rot = pg.transform.rotozoom(self.img, self.angle, 1.0)
            pos = rot.get_rect(center=self.rect.center)
            surface.blit(rot, pos.topleft)
        else:
            pg.draw.ellipse(surface, (200, 50, 50), self.rect)

# -------------------------
# Game class
# -------------------------
class SpaceGame:
    def __init__(self):
        pg.init()
        pg.font.init()

        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Space Invaders — A* Auto-Avoid")
        self.clock = pg.time.Clock()
        self.running = True

        self.bg = safe_image_load(["background.jpg", "background.png", "Spacegame.jfif"], size=(WIDTH, HEIGHT))
        self.player_img = safe_image_load(["player.png", "spaceship3.png"], size=(PLAYER_W, PLAYER_H))
        self.enemy_img = safe_image_load(["enemy.png", "asteroid.png"], size=(ENEMY_W, ENEMY_H))
        self.bullet_img = safe_image_load(["bullet.png", "laser.png"], size=(BULLET_W, BULLET_H))

        self.state = "menu"
        self.score = 0
        self.start_time = 0
        self.player = pg.Rect(WIDTH//2 - PLAYER_W//2, HEIGHT - PLAYER_H - 10, PLAYER_W, PLAYER_H)
        self.bullets = []
        self.enemies = []
        self.last_shot = 0
        self.last_enemy_spawn = 0

        self.font = pg.font.SysFont("arial", 24)
        self.big_font = pg.font.SysFont("arial", 56, bold=True)
        self.small_font = pg.font.SysFont("arial", 18)

        self.play_btn = pg.Rect(WIDTH//2 - 120, HEIGHT//2 - 40, 240, 60)
        self.quit_btn = pg.Rect(WIDTH//2 - 120, HEIGHT//2 + 30, 240, 60)
        self.restart_btn = pg.Rect(WIDTH//2 - 120, HEIGHT//2 + 40, 240, 60)

        # A* tuning (survival-focused)
        self.search_depth = 30         # frames to look ahead
        self.max_nodes = 4000         # node visit cap
        self.move_cost = 1.0          # penalty per pixel moved (reduces jitter)
        self.danger_scale = 1.0       # scales danger penalties
        self.collision_penalty = 1e6  # extremely large cost for collision (avoid at all costs)

    def reset_game(self):
        self.score = 0
        self.player.x = WIDTH//2 - PLAYER_W//2
        self.player.y = HEIGHT - PLAYER_H - 10
        self.bullets.clear()
        self.enemies.clear()
        self.last_shot = 0
        self.last_enemy_spawn = pg.time.get_ticks()
        self.start_time = time.time()

    def spawn_enemy_wave(self, count=1):
        for _ in range(count):
            x = random.randint(0, WIDTH - ENEMY_W)
            self.enemies.append(Enemy(x, -ENEMY_H - random.randint(0, 200), img=self.enemy_img))

    def shoot(self):
        now = pg.time.get_ticks()
        if now - self.last_shot >= SHOOT_COOLDOWN:
            x = self.player.centerx - BULLET_W//2
            y = self.player.top - BULLET_H
            self.bullets.append(Bullet(x, y, img=self.bullet_img))
            self.last_shot = now

    # -------------------------
    # A* autopilot (survival-focused)
    # State: (rounded_player_x, step)
    # Moves: -PLAYER_VEL, 0, +PLAYER_VEL
    # Cost: movement cost + danger cost; collision => huge penalty
    # Heuristic: minimal vertical distance to nearest enemy minus horizontal distance (lower => more urgent)
    # -------------------------
    def plan_move_astar(self, depth=None):
        if depth is None:
            depth = self.search_depth

        if not self.enemies:
            return 0

        start_x = int(self.player.x)
        pw = self.player.width
        ph = self.player.height

        # initial enemy rect copies
        init_enemies = [e.rect.copy() for e in self.enemies]

        def collide(player_x, enemies):
            pr = pg.Rect(int(player_x), self.player.y, pw, ph)
            for er in enemies:
                if pr.colliderect(er):
                    return True
            return False

        # if already colliding, try immediate escape moves
        if collide(start_x, init_enemies):
            for mv in [-PLAYER_VEL, PLAYER_VEL, 0]:
                nx = max(0, min(WIDTH - pw, start_x + mv))
                if not collide(nx, init_enemies):
                    return nx - start_x

        # A* structures
        # priority queue items: (f_score, g_cost, player_x, step, path_first_move, enemies_positions_list)
        pq = []
        visited = dict()  # (x, step) -> best_g_cost
        nodes = 0

        # heuristic function: estimate danger remaining (lower is better)
        def heuristic(player_x, enemies, step_remaining):
            # look at the nearest enemy in vertical dimension (that will reach player sooner)
            min_val = float('inf')
            px_center = player_x + pw / 2
            for er in enemies:
                # vertical distance from enemy bottom to player's top
                vdist = (self.player.y - er.y)
                if vdist < -ENEMY_H:  # enemy already passed far below
                    continue
                # horizontal distance between centers
                hdist = abs(er.centerx - px_center)
                # create a score: smaller vertical + small horizontal -> more dangerous
                # scale so nearer enemies produce lower heuristic (urgency)
                val = vdist + 0.5 * hdist
                if val < min_val:
                    min_val = val
            if min_val == float('inf'):
                return 0
            # clamp and invert so lower distance => higher heuristic urgency; but A* expects admissible heuristic
            # We return max(0, min_val) as a simple optimistic estimate of how far until danger reduces
            return max(0, min_val) / (ENEMY_VEL + 1.0)

        # initial node
        h0 = heuristic(start_x, init_enemies, depth)
        heapq.heappush(pq, (h0, 0.0, start_x, 0, None, init_enemies))
        visited[(start_x, 0)] = 0.0

        best_path_first_move = 0  # fallback
        best_safe = None

        while pq:
            nodes += 1
            if nodes > self.max_nodes:
                break
            f, g, px, step, first_move, enemies_sim = heapq.heappop(pq)

            # if this node is at collision -> skip
            if collide(px, enemies_sim):
                continue

            # if reached required depth without collisions -> choose path
            if step >= depth:
                if first_move is None:
                    return 0
                return first_move

            # expand neighbors (prioritize stay to reduce jitter)
            for mv in [0, -PLAYER_VEL, PLAYER_VEL]:
                nx = px + mv
                nx = max(0, min(WIDTH - pw, nx))
                # simulate enemies moving one frame down (for the next step)
                next_enemies = [er.copy() for er in enemies_sim]
                for er in next_enemies:
                    er.y += ENEMY_VEL

                # collision check after move and enemy advance
                if collide(nx, next_enemies):
                    # extremely bad, skip adding but we might consider for fallback
                    continue

                # movement cost (penalize big moves)
                move_pen = self.move_cost * abs(nx - px)

                # danger cost: sum of penalties based on proximity (vertical closeness weighted by horizontal closeness)
                danger = 0.0
                px_center = nx + pw / 2
                for er in next_enemies:
                    vdist = (self.player.y - er.y)
                    hdist = abs(er.centerx - px_center)
                    if vdist <= 0:
                        # enemy is at or below player — not an immediate vertical threat
                        continue
                    # horizontal influence: strong if overlapping horizontally
                    horiz_factor = max(0.0, (ENEMY_W/2 + pw/2) - hdist)
                    # vertical urgency: smaller vertical distance => higher danger
                    vert_factor = max(0.0, (200 - vdist))  # scale: enemies within 200px above are concerning
                    # combine
                    danger += (vert_factor * (1.0 + horiz_factor / (ENEMY_W/2 + pw/2)))
                danger = danger * self.danger_scale / 50.0  # normalize

                step_cost = move_pen + danger

                new_g = g + step_cost

                state_key = (int(nx), step + 1)
                prev_g = visited.get(state_key, float('inf'))
                if new_g + 1e-9 < prev_g:
                    visited[state_key] = new_g
                    h = heuristic(nx, next_enemies, depth - (step + 1))
                    fscore = new_g + h
                    new_first = mv if first_move is None else first_move
                    heapq.heappush(pq, (fscore, new_g, nx, step + 1, new_first, next_enemies))
                    best_path_first_move = new_first

        # fallback: if A* found nothing (rare), pick safest among immediate moves
        best_move = 0
        best_score = -float('inf')
        for mv in [0, -PLAYER_VEL, PLAYER_VEL]:
            nx = max(0, min(WIDTH - pw, start_x + mv))
            # evaluate min vertical distance adjusted by horizontal overlap
            min_dist = float('inf')
            px_center = nx + pw/2
            for er in init_enemies:
                vdist = (self.player.y - er.y)
                hdist = abs(er.centerx - px_center)
                score = vdist - 0.5 * hdist
                if score < min_dist:
                    min_dist = score
            # larger min_dist is better (further away)
            if min_dist > best_score:
                best_score = min_dist
                best_move = mv

        return best_move

    def update_playing(self):
        now = pg.time.get_ticks()

        if now - self.last_enemy_spawn >= ENEMY_SPAWN_INTERVAL:
            extra = min(4, 1 + self.score // 10)
            self.spawn_enemy_wave(1 + random.randint(0, extra))
            self.last_enemy_spawn = now

        move_dx = self.plan_move_astar()
        if move_dx != 0:
            self.player.x += int(move_dx)
        # keep player within bounds
        self.player.x = max(0, min(WIDTH - self.player.width, self.player.x))

        # keyboard still allowed for shooting and toggles
        keys = pg.key.get_pressed()
        if keys[pg.K_SPACE]:
            self.shoot()

        for b in self.bullets[:]:
            b.update()
            if b.rect.bottom < 0:
                self.bullets.remove(b)

        for e in self.enemies[:]:
            e.update()
            if e.rect.top > HEIGHT:
                self.enemies.remove(e)
            if e.rect.colliderect(self.player):
                self.state = "game_over"
                return

        for b in self.bullets[:]:
            for e in self.enemies[:]:
                if b.rect.colliderect(e.rect):
                    try:
                        self.bullets.remove(b)
                    except ValueError:
                        pass
                    try:
                        self.enemies.remove(e)
                    except ValueError:
                        pass
                    self.score += 1
                    break

        self.draw_playing()

    def draw_playing(self):
        if self.bg:
            self.screen.blit(self.bg, (0, 0))
        else:
            self.screen.fill((10, 10, 30))

        if self.player_img:
            self.screen.blit(self.player_img, (self.player.x, self.player.y))
        else:
            pg.draw.rect(self.screen, (0, 200, 255), self.player)

        for b in self.bullets:
            b.draw(self.screen)

        for e in self.enemies:
            e.draw(self.screen)

        elapsed = int(time.time() - self.start_time)
        self.screen.blit(self.font.render(f"Score: {self.score}", True, (255,255,255)), (10, 10))
        self.screen.blit(self.font.render(f"Time: {elapsed}s", True, (255,255,255)), (10, 40))

        pg.display.flip()

    def draw_menu(self):
        if self.bg:
            self.screen.blit(self.bg, (0,0))
            overlay = pg.Surface((WIDTH,HEIGHT), pg.SRCALPHA)
            overlay.fill((0,0,0,160))
            self.screen.blit(overlay, (0,0))
        else:
            self.screen.fill((0,0,40))

        title = self.big_font.render("SPACE INVADERS", True, (255,255,255))
        self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 120))

        pg.draw.rect(self.screen, (80,160,255), self.play_btn, border_radius=10)
        pg.draw.rect(self.screen, (255,100,100), self.quit_btn, border_radius=10)

        self.screen.blit(self.font.render("Play", True, (0,0,0)),
                         (self.play_btn.centerx - 25, self.play_btn.centery - 12))
        self.screen.blit(self.font.render("Quit", True, (0,0,0)),
                         (self.quit_btn.centerx - 25, self.quit_btn.centery - 12))

        pg.display.flip()

    def draw_game_over(self):
        if self.bg:
            self.screen.blit(self.bg, (0,0))
            overlay = pg.Surface((WIDTH,HEIGHT), pg.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
        else:
            self.screen.fill((20,0,0))

        over = self.big_font.render("GAME OVER", True, (255,50,50))
        score_tx = self.font.render(f"Your Score: {self.score}", True, (255,255,255))
        self.screen.blit(over, (WIDTH//2 - over.get_width()//2, 180))
        self.screen.blit(score_tx, (WIDTH//2 - score_tx.get_width()//2, 260))

        pg.draw.rect(self.screen, (80,160,255), self.restart_btn, border_radius=10)
        self.screen.blit(self.font.render("Restart", True, (0,0,0)),
                         (self.restart_btn.centerx - 40, self.restart_btn.centery - 12))

        quit_rect = pg.Rect(WIDTH//2 - 120, self.restart_btn.bottom + 20, 240, 50)
        pg.draw.rect(self.screen, (255,100,100), quit_rect, border_radius=10)
        self.screen.blit(self.font.render("Quit", True, (0,0,0)),
                         (quit_rect.centerx - 20, quit_rect.centery - 12))

        pg.display.flip()
        return quit_rect

    def run(self):

        while self.running:
            self.clock.tick(FPS)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                    break

                if event.type == pg.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if self.state == "menu":
                        if self.play_btn.collidepoint(mx, my):
                            self.state = "playing"
                            self.reset_game()
                        if self.quit_btn.collidepoint(mx, my):
                            self.running = False

                    if self.state == "game_over":
                        if self.restart_btn.collidepoint(mx, my):
                            self.state = "playing"
                            self.reset_game()
                        elif self.quit_game_rect.collidepoint(mx, my):
                            self.running = False

                if event.type == pg.KEYDOWN:
                    if self.state == "menu" and event.key == pg.K_RETURN:
                        self.state = "playing"
                        self.reset_game()

                    if self.state == "playing" and event.key == pg.K_SPACE:
                        self.shoot()

                    if self.state == "game_over" and event.key == pg.K_RETURN:
                        self.state = "playing"
                        self.reset_game()

            if self.state == "menu":
                self.draw_menu()
            elif self.state == "playing":
                self.update_playing()
            elif self.state == "game_over":
                quit_rect = self.draw_game_over()
                self.quit_game_rect = quit_rect

                if pg.key.get_pressed()[pg.K_q]:
                    self.running = False

        pg.quit()

if __name__ == "__main__":
    SpaceGame().run()
