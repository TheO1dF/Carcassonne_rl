import subprocess
import time

# 3位世界冠军的玩家 ID
CHAMPIONS = ["90101428", "90314669", "90839284"]

# 每位玩家下载 150 局
LIMIT = 150 

def main():
    print("🌙 开始夜间挂机下载模式...")
    print(f"🏆 目标玩家: {CHAMPIONS}")
    print(f"📊 每人最大下载局数: {LIMIT}")
    
    for player in CHAMPIONS:
        print(f"\n{'='*50}")
        print(f"🚀 开始下载世界冠军 {player} 的录像...")
        print(f"{'='*50}\n")
        
        # 调用我们刚刚优化好的极速爬虫脚本
        command = ["python", "bga_scraper.py", "--player", player, "--limit", str(LIMIT)]
        result = subprocess.run(command)
        
        if result.returncode != 0:
            print(f"\n⚠️ 警告: 玩家 {player} 的下载过程似乎遇到了意外中断。")
        else:
            print(f"\n✅ 玩家 {player} 的任务已圆满完成！")
            
        # 爬完一个大神后，休息 15 秒，保护账号
        if player != CHAMPIONS[-1]:
            print("💤 休息 15 秒后继续下一个大神的下载...")
            time.sleep(15)

    print("\n🎉 所有挂机任务全部完成！安心睡觉吧，明天见！")

if __name__ == "__main__":
    main()