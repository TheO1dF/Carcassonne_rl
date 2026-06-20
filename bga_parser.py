import json

JSON_FILE = "bga_replay.json"

def parse_bga_log(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    # 提取核心数据包
    packets = log_data["data"]["data"]
    
    # 用一个列表来按顺序存储所有的动作
    actions = []
    
    for packet in packets:
        # 有些包可能没有 "data" 字段，安全过滤
        if "data" not in packet:
            continue
            
        for item in packet["data"]:
            event_type = item.get("type")
            
            if event_type == "playTile":
                args = item["args"]
                # 记录板块放置
                actions.append({
                    "player": args["player_name"],
                    "tile_instance_id": str(args["id"]), # 唯一ID，用于绑定米宝
                    "tile_type": str(args["type"]),      # BGA内部的板块种类(1-24)
                    "x": int(args["x"]) + 15,            # 转换到 env.py 的中心坐标
                    "y": int(args["y"]) + 15,
                    "rot": int(args["ori"]),
                    "meeple_target": None,               # 默认无米宝
                    "meeple_pos": None
                })
                
            elif event_type == "playPartisan":
                args = item["args"]
                partisan_tile_id = str(args["id"])
                
                # 倒序查找刚刚放下的那块牌，把米宝信息补充进去
                for action in reversed(actions):
                    if action["tile_instance_id"] == partisan_tile_id:
                        action["meeple_target"] = args["target"] # e.g. "city", "road"
                        action["meeple_pos"] = str(args["pos"])
                        break
                        
    # 打印最终的完美动作序列
    print("--- 🎬 BGA 完美对局动作解析 ---")
    for i, a in enumerate(actions):
        if a['meeple_target']:
            meeple_str = f"👉放米宝: [{a['meeple_target']:<5}] (BGA位置:{a['meeple_pos']})"
        else:
            meeple_str = "无米宝"
            
        print(f"步数 {i+1:2d} | 玩家: {a['player']:<9} | 坐标: ({a['x']:2d}, {a['y']:2d}) | "
              f"BGA板块: {a['tile_type']:<2} | 朝向: {a['rot']} | {meeple_str}")

if __name__ == "__main__":
    parse_bga_log(JSON_FILE)