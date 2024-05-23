import pandas as pd
import matplotlib.pyplot as plt
import cv2


class Visualizer():
    def __init__(self):
        # 列が1から18番までの馬のx，y座標を格納するDataFrameを作成
        # columns = [i for i in range(1, 19)]
        columns = [f"{i}_x" if j == 0 else f"{i}_y" for i in range(1, 19) for j in range(2)]
        self.df = pd.DataFrame(columns=columns)

        self.tracker = cv2.legacy.MultiTracker_create()
    
    def add_location(self, horse, now_location, now_frame):
        now_x, now_y = now_location
        self.df.loc[now_frame, f'{horse.number}_x'] = now_x
        self.df.loc[now_frame, f'{horse.number}_y'] = now_y

    def init_plot(self):
        # 描画の設定
        plt.ion()  # 対話モードをオンにする
        self.fig, self.ax = plt.subplots()

    def visualize(self, horse_num):
        # プロットデータを更新（ここではランダムデータを使用）
        self.ax.plot(self.df[f'{horse_num}_x'], self.df[f'{horse_num}_y'], label=f'Horse {horse_num}')    #TODO 全部の馬に対してプロットする

    def show(self):
        plt.draw()
        plt.pause(0.001)  # 更新間隔
