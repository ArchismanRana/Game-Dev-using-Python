# space_invaders_mediapipe.py
import os
import random
import time
import pygame as pg
import cv2
import mediapipe as mp

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

# Enemy
ENEMY_W, ENEMY_H = 48, 48
ENEMY_VEL = 3
ENEMY_SPAWN_INTERVAL = 1200  # ms
ENEMY_ANIM_ROT_SPEED = 2  # degrees per frame

CAM_WIDTH = 640
CAM_HEIGHT = 480
SHOW_CAM = False

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
        pg.display.set_caption("Space Invaders — Hand Control")
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

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )

        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        self.play_btn = pg.Rect(WIDTH//2 - 120, HEIGHT//2 - 40, 240, 60)
        self.quit_btn = pg.Rect(WIDTH//2 - 120, HEIGHT//2 + 30, 240, 60)
        self.restart_btn = pg.Rect(WIDTH//2 - 120, HEIGHT//2 + 40, 240, 60)

        self.show_cam_in_game = True

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

    def handle_camera(self):
        if not self.cap.isOpened():
            return None, None, False  # no camera, no shoot

        ret, frame = self.cap.read()
        if not ret:
            return None, frame, False

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        index_x_norm = None
        shoot_gesture = False

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]

            # ----- Finger landmarks -----
            ix = lm.landmark[8].x  # index tip
            iy = lm.landmark[8].y
            tx = lm.landmark[4].x  # thumb tip
            ty = lm.landmark[4].y

            # movement
            index_x_norm = ix

            # ----- PINCH DETECTION -----
            dist = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5

            if dist < 0.05:  # pinch threshold
                shoot_gesture = True

            # draw hand landmarks
            if SHOW_CAM or self.show_cam_in_game:
                mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)

        return index_x_norm, frame, shoot_gesture

    def shoot(self):
        now = pg.time.get_ticks()
        if now - self.last_shot >= SHOOT_COOLDOWN:
            x = self.player.centerx - BULLET_W//2
            y = self.player.top - BULLET_H
            self.bullets.append(Bullet(x, y, img=self.bullet_img))
            self.last_shot = now

    def update_playing(self):
        now = pg.time.get_ticks()

        if now - self.last_enemy_spawn >= ENEMY_SPAWN_INTERVAL:
            extra = min(4, 1 + self.score // 10)
            self.spawn_enemy_wave(1 + random.randint(0, extra))
            self.last_enemy_spawn = now

        index_x_norm, cam_frame, shoot_gesture = self.handle_camera()
        if index_x_norm is not None:
            target_x = int(index_x_norm * WIDTH) - self.player.width // 2
            target_x = max(0, min(WIDTH - self.player.width, target_x))
            self.player.x = int(self.player.x * 0.7 + target_x * 0.3)

        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            self.player.x -= PLAYER_VEL
        if keys[pg.K_RIGHT]:
            self.player.x += PLAYER_VEL
        if shoot_gesture:
            self.shoot()

        self.player.x = max(0, min(WIDTH - self.player.width, self.player.x))

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
                    self.bullets.remove(b)
                    self.enemies.remove(e)
                    self.score += 1
                    break

        self.draw_playing(cam_frame)

    def draw_playing(self, cam_frame=None):
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

        if self.show_cam_in_game and cam_frame is not None:
            frame_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (160, 120))
            surf = pg.surfarray.make_surface(frame_rgb.swapaxes(0,1))
            self.screen.blit(surf, (WIDTH - 170, 10))

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

                    if event.key == pg.K_m:
                        self.show_cam_in_game = not self.show_cam_in_game

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

        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        pg.quit()

if __name__ == "__main__":
    SpaceGame().run()
