"""
tile_mapper_gui.py
==================
用于人工视觉校准 BGA 版块与本地版块朝向的工具。
操作说明：
- 左右方向键（Left/Right）：旋转当前版块。
- 回车键（Enter）：确认当前版块的旋转角度，并进入下一块。
- 退格键（Backspace）：返回上一个版块重新校准。
"""

import pygame
import sys
import json
from tiles import TILE_TYPES, PlacedTile, N, E, S, W

# --------------------------------------------------------------------------- #
# BGA 到本地的映射字典 (根据你之前提供的正确映射)
# --------------------------------------------------------------------------- #
BGA_TO_LOCAL = {
    "1":"T", "2":"S", "3":"N", "4":"M", "5":"P", "6":"O", "7":"G", "8":"F", 
    "9":"I", "10":"H", "11":"E", "12":"K", "13":"J", "14":"L", "15":"D", "16":"U", 
    "17":"V", "18":"W", "19":"X", "20":"B", "21":"A", "22":"C", "23":"R", "24":"Q"
}

# 按 BGA ID 排序，方便你对照着 BGA 图鉴一个个看
ORDERED_TILES = sorted(BGA_TO_LOCAL.items(), key=lambda x: int(x[0]))

# --------------------------------------------------------------------------- #
# 渲染参数 (直接复用 play_gui.py)
# --------------------------------------------------------------------------- #
FIELD_GREEN = (108, 168, 86)
CITY_BROWN = (158, 102, 52)
CITY_OUTLINE = (104, 62, 28)
ROAD_GRAY = (225, 225, 222)
ROAD_OUTLINE = (120, 120, 118)
SHIELD_COL = (224, 226, 238)
GRID_LINE = (52, 60, 52)

CITY_POLY = {
    N: [(0, 0), (1, 0), (0.82, 0.45), (0.18, 0.45)],
    E: [(1, 0), (1, 1), (0.55, 0.82), (0.55, 0.18)],
    S: [(0, 1), (1, 1), (0.82, 0.55), (0.18, 0.55)],
    W: [(0, 0), (0, 1), (0.45, 0.82), (0.45, 0.18)],
}
EDGE_MID = {N: (0.5, 0.0), E: (1.0, 0.5), S: (0.5, 1.0), W: (0.0, 0.5)}
CENTER = (0.5, 0.5)

def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

def draw_tile(surface, tile_type, rotation, size):
    surface.fill(FIELD_GREEN)
    placed = PlacedTile(tile_type, rotation)

    # 城市
    for edges, shield in placed.cities:
        for e in edges:
            poly = [(p[0] * size, p[1] * size) for p in CITY_POLY[e]]
            pygame.draw.polygon(surface, CITY_BROWN, poly)
            pygame.draw.polygon(surface, CITY_OUTLINE, poly, max(1, size // 36))
        if shield:
            ex = next(iter(edges))
            cx, cy = _lerp(EDGE_MID[ex], CENTER, 0.45)
            s = max(4, int(size * 0.13))
            r = pygame.Rect(0, 0, s, s)
            r.center = (cx * size, cy * size)
            pygame.draw.rect(surface, SHIELD_COL, r)
            pygame.draw.rect(surface, CITY_OUTLINE, r, 1)

    # 道路
    rw = max(2, int(size * 0.13))
    for edges in placed.roads:
        for e in edges:
            a = (EDGE_MID[e][0] * size, EDGE_MID[e][1] * size)
            b = (CENTER[0] * size, CENTER[1] * size)
            pygame.draw.line(surface, ROAD_OUTLINE, a, b, rw + 2)
            pygame.draw.line(surface, ROAD_GRAY, a, b, rw)
    if placed.roads:
        pygame.draw.circle(surface, ROAD_GRAY, (int(size * 0.5), int(size * 0.5)), max(2, rw // 2))

    # 修道院
    if placed.monastery:
        cx, cy = size * 0.5, size * 0.54
        w = size * 0.26
        body = pygame.Rect(0, 0, w, w * 0.7)
        body.center = (cx, cy + w * 0.12)
        pygame.draw.rect(surface, (228, 204, 162), body)
        pygame.draw.rect(surface, (120, 82, 40), body, max(1, size // 50))
        roof = [(cx - w * 0.6, body.top), (cx + w * 0.6, body.top), (cx, body.top - w * 0.55)]
        pygame.draw.polygon(surface, (164, 64, 44), roof)

    pygame.draw.rect(surface, GRID_LINE, surface.get_rect(), 2)

# --------------------------------------------------------------------------- #
# 主程序
# --------------------------------------------------------------------------- #
def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("BGA 图像校准工具")
    font = pygame.font.SysFont("consolas", 24)
    font_large = pygame.font.SysFont("consolas", 36, bold=True)
    clock = pygame.time.Clock()

    current_idx = 0
    offsets = {local_id: 0 for _, local_id in ORDERED_TILES}
    
    running = True
    done = False

    while running:
        screen.fill((30, 33, 42))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN and not done:
                bga_id, local_id = ORDERED_TILES[current_idx]
                
                if event.key == pygame.K_RIGHT:
                    offsets[local_id] = (offsets[local_id] + 1) % 4
                elif event.key == pygame.K_LEFT:
                    offsets[local_id] = (offsets[local_id] - 1) % 4
                elif event.key == pygame.K_RETURN:
                    current_idx += 1
                    if current_idx >= len(ORDERED_TILES):
                        done = True
                elif event.key == pygame.K_BACKSPACE:
                    if current_idx > 0:
                        current_idx -= 1

        if not done:
            bga_id, local_id = ORDERED_TILES[current_idx]
            current_rot = offsets[local_id]
            tile_type = TILE_TYPES[local_id]

            # 绘制信息
            text1 = font_large.render(f"进度: {current_idx + 1} / 24", True, (255, 255, 255))
            text2 = font_large.render(f"正在校准 BGA ID: {bga_id} (本地字母: {local_id})", True, (248, 240, 130))
            text3 = font.render(f"当前偏移量: {current_rot} (顺时针旋转 {current_rot * 90} 度)", True, (200, 200, 200))
            text4 = font.render("[←][→] 旋转版块  |  [Enter] 确认并下一步  |  [Backspace] 上一步", True, (150, 150, 150))

            screen.blit(text1, (50, 30))
            screen.blit(text2, (50, 80))
            screen.blit(text3, (50, 130))
            screen.blit(text4, (50, 530))

            # 绘制中间大版块
            tile_size = 300
            tile_surf = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
            draw_tile(tile_surf, tile_type, current_rot, tile_size)
            screen.blit(tile_surf, (250, 200))

        else:
            # 完成后输出结果
            success_txt = font_large.render("校准完成！结果已打印到控制台。", True, (108, 168, 86))
            screen.blit(success_txt, (150, 250))
            pygame.display.flip()
            
            print("\n" + "="*50)
            print("请复制以下字典到你的 bga_translator.py 中，替换原有的 IMAGE_OFFSETS：\n")
            print("IMAGE_OFFSETS = {")
            
            # 按字母 A-X 排序打印
            sorted_offsets = dict(sorted(offsets.items()))
            items = [f'"{k}": {v}' for k, v in sorted_offsets.items()]
            
            # 格式化打印为多行
            for i in range(0, len(items), 8):
                print("    " + ", ".join(items[i:i+8]) + ("," if i+8 < len(items) else ""))
            print("}")
            print("="*50 + "\n")
            
            pygame.time.wait(3000)
            running = False

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()