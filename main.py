import cv2

from video import Video
from frame import Frame
from horse import Horse
from visualizer import Visualizer

youtube_path = "2024年 京王杯スプリングカップ（GⅡ）  ウインマーベル  JRA公式.mp4"
skip_frame = 600
template_size = (35, 35)

if __name__ == "__main__":
    youtube_path = "2024年 京王杯スプリングカップ（GⅡ）  ウインマーベル  JRA公式.mp4"
    keio_spring_cup = Video(youtube_path)
    v = Visualizer()

    horse8 = Horse(8, "red")

    for _ in range(skip_frame):
        ret, frame = keio_spring_cup.cap.read()
    
    f = Frame(frame)
    horese8_left = f.template_matching("./template/i8.png")
    horse8_bbox = (horese8_left[0], horese8_left[1], horese8_left[0]+template_size[1], horese8_left[1]+template_size[0])

    # v.tracker.add(cv2.legacy.TrackerKCF_create(), frame, horse8_bbox)
    # v.tracker.add(cv2.legacy.TrackerCSRT_create(), frame, horse8_bbox)

    for num_frame in range(keio_spring_cup.frame_count):
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
        ret, frame = keio_spring_cup.cap.read()

        # フレームが正しく読み込めなかった場合、ループを抜ける
        if not ret:
            print("フレームを読み込めませんでした。動画の終了またはエラーです。")
            break
        
        
        f = Frame(frame)
        horese8_left = f.template_matching("./template/i8.png", horse8_bbox)
        horse8_bbox = (horese8_left[0], horese8_left[1], template_size[1], template_size[0])
        # circles = f.draw_circles(f.detect_circles())     
        # パターンマッチング
        # horese8_bbox = f.pattern_matching("./template/i8.png")
        # トラッキング
        # ok, bboxes = v.tracker.update(frame)
        # template_sizeを使ってbboxesを修正
        # fixed_bboxes = [(bbox[0], bbox[1], template_size[1], template_size[0]) for bbox in bboxes]

        # race.dfの一番下の行に8番の馬の座標を追加
        # v.ax.clear()    # グラフをクリア
        # v.add_location(horse8, horese8_bbox, num_frame)
        # v.visualize(horse8.number)

        
        # v.show()

        


        f.draw_rectangle(horse8_bbox)
        
        # RGB画像として表示
        cv2.imshow('Frame', frame)


        # 'q'キーが押されたらループから抜ける
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(v.df)
            break

