import cv2
import numpy as np
import matplotlib.pyplot as plt
from frame import Frame
from horse import Horse, Race

class Video:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)   # OpenCVを使用して動画を読み込む
        # すべてのフレーム数を取得
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) # 動画のフレームレートを取得

    def play(self):
        # print(f'Playing {self.title} for {self.duration} seconds')
        while True:
            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
            ret, frame = self.cap.read()

            # フレームが正しく読み込めなかった場合、ループを抜ける
            if not ret:
                print("フレームを読み込めませんでした。動画の終了またはエラーです。")
                break

            # フレームを表示
            height, width, _ = frame.shape

            # 指定された領域を長方形で囲む
            top_left = (int(width*0.18), int(height*0.75))
            bottom_right = (int(width*0.82), height)
            
            
            f = Frame(frame)
            circles = f.draw_circles(f.detect_circles())     

            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 3)  # 青色で3ピクセルの線幅
            cv2.imshow('Frame', frame)
    

            # 'q'キーが押されたらループから抜ける
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    def play_with_plt(self):
        wait_time = int(1000 / self.fps)

        # プロットを設定
        plt.ion()  # 対話モードをオンにする
        fig, ax = plt.subplots()

        data_list = []
        # 動画をフレームごとに表示
        while True:
            ret, frame = self.cap.read()

            # フレームが正しく読み込めなかった場合、ループを抜ける
            if not ret:
                print("フレームを読み込めませんでした。動画の終了またはエラーです。")
                break

            # OpenCVウィンドウでフレームを表示
            cv2.imshow('Video Frame', frame)

            # フレームごとに馬の座標を検出
            # # horese_list = []
            # horese_df = Race()
            # for circle in circles:
            #     x, y, _ = circle
                # horese_num = f.detect_number(f.detect_circles())  # 馬番号を特定
            #     # horese_list.append(Horse(horese_num, x, y, "red"))    # 色を入れる
           

            # 'q'キーが押されたらループから抜ける
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

        # リソースを解放
        self.cap.release()
        cv2.destroyAllWindows()
        plt.close()
    
    def __str__(self):
        return self.title

    
if __name__ == "__main__":
    youtube_path = "2024年 京王杯スプリングカップ（GⅡ）  ウインマーベル  JRA公式.mp4"
    keio_spring_cup = Video(youtube_path)
    keio_spring_cup.play()
    # keio_spring_cup.play_with_plt()