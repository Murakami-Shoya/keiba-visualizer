import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import pandas as pd

class Frame:
    """
    馬の情報を認識して表示するためのクラス
    input: image
    """
    def __init__(self, image):
        self.image = image
        # DataFrameを作成
        self.df = pd.DataFrame(columns=['x', 'y', 'horse_num'])

        # 画像をグレースケールに変換
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 探索範囲を指定
        height, width, _ = image.shape
        # 
        # ノイズを除去するために画像をぼかす
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0) # 引数：画像、カーネルサイズ、標準偏差


    def detect_circles(self):
        height, width, _ = self.image.shape
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # heiht, width でマスク
        mask = np.zeros_like(gray)
        mask[int(height*0.75):height, int(width*0.18):int(width*0.82)] = 255
        masked_blurred = cv2.bitwise_and(gray, gray, mask=mask)
        circles = cv2.HoughCircles(masked_blurred, cv2.HOUGH_GRADIENT,
                                dp=1, minDist=5, param1=50, param2=19, minRadius=13, maxRadius=20)
        return np.round(circles[0, :]).astype("int")


    def detect_number(self, circle, mask_r=17):
        (x, y, r) = circle
        roi = self.gray[y-r:y+r, x-r:x+r]
        # 半径rの円でimgをマスク
        mask = np.zeros_like(roi)
        cv2.circle(mask, (r, r), int(mask_r/1.2), 255, -1)  # 引数：画像、中心座標、半径、色、塗りつぶし

        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        # マスクした外側を中央値の色で塗りつぶす
        masked_roi[mask == 0] = np.median(masked_roi[mask != 0])
        
        plt.imshow(masked_roi, cmap='gray')
        plt.show()
        
        # 数字認識のための前処理
        blurred = cv2.GaussianBlur(masked_roi, (5, 5), 0)   # ガウシアンフィルタ
        _, roi_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # 2値化
        # ROIの表示
        # plt.imshow(roi_thresh, cmap='gray')
        # plt.show()
        # 数字認識 (ここでは例としてpytesseractを使用)
        text = pytesseract.image_to_string(roi_thresh, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789') # 引数：画像、認識モード、OCRエンジン、認識対象文字
        try:
            detected_number = int(text.strip())
        except ValueError:
            print("認識されたテキストが数字ではありません。")
            detected_number = -1  # 数字が認識されなかった場合のデフォルト値
        
        return detected_number

    # def detect_horses(self, circles):
    #     for circle in circles[0]:
    #         x, y, r = circle
    #         # 馬番号を認識
    #         horse_num = self.detect_number(circle, self.image)
    #         self.df = self.df.append({'x': x, 'y': y, 'horse_num': horse_num}, ignore_index=True)
    #     return self.df
    
    def draw_circles(self, circles):
        # 検出された円を描画
        if circles is not None:
            # circles = np.uint16(np.around(circles))
            for i in circles:
                cv2.circle(self.image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(self.image, (i[0], i[1]), 2, (0, 0, 255), 3) # 中心点を描画
        # RGB画像を返す
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def pattern_matching(self, template_path):
        # 画像を読み込む
        template = cv2.imread(template_path, 0)  # テンプレートはグレースケールで読み込む

        # roi = self.gray[int(height*0.75):height, int(width*0.18):int(width*0.82)]

        # テンプレートマッチングを実行
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)

        # 最も一致する位置を取得
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])


if __name__ == "__main__":
    image = cv2.imread("frame_500.jpg")
    frame = Frame(image)
    circles = frame.detect_circles()
    image_with_circles = frame.draw_circles(circles)
    plt.imshow(image_with_circles)
    plt.show()
    # df = visualizer.detect_horses(circles)
    # print(df)