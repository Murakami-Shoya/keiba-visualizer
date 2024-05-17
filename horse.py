import pandas as pd

class Horse():
    def __init__(self, number, x, y, color):
        self.number = number
        self.x = x
        self.y = y
        self.color = color

class Race():
    def __init__(self):
        # 列が1から18番までの馬のx，y座標を格納するDataFrameを作成
        columns = [i for i in range(1, 19)]
        self.df = pd.DataFrame(columns=columns)
    
    def add_horse(self, horse):
        self.df[horse.number] = [horse.x, horse.y]

    def visualize(self):
        # プロットデータを更新（ここではランダムデータを使用）
        data = np.random.rand(1)   # ランダムデータを生成
        data_list.append(data)
        ax.clear()
        ax.plot(data_list)
        plt.draw()
        plt.pause(0.001)  # 更新間隔