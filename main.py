import cv2

from video import Video

youtube_path = "2024年 京王杯スプリングカップ（GⅡ）  ウインマーベル  JRA公式.mp4"
skip_frame = 400

if __name__ == "__main__":
    keio_spring_cup = Video(youtube_path)
    # keio_spring_cup.play_with_plt()

    # # skip_frameフレーム目までスキップ
    # for i in range(frame_count):
    #     ret, frame = cap.read()
        
    #     if i <= skip_frame - 1:
    #         continue
    #     else:
    #         # 1フレームごと馬の座標を検出
    #             # フレームを表示
    #         cv2.imshow('Frame', frame)

    #         # 'q'キーが押されたらループから抜ける
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break


