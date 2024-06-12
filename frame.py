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
        self.height, self.width, _ = image.shape
        self.left = int(self.width*0.18)
        self.top = int(self.height*0.75)
        self.right = int(self.width*0.82)
        self.masked_gray = self.gray[self.top:self.height, self.left:self.right]
        # self.masked_binary = cv2.threshold(self.masked_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]    # 2値化
        # ノイズを除去するために画像をぼかす
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0) # 引数：画像、カーネルサイズ、標準偏差
        
        # templateの大きさ
        self.template_shape = None

    # 円検出
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
    
    # def draw_circles(self, circles):
    #     # 検出された円を描画
    #     if circles is not None:
    #         # circles = np.uint16(np.around(circles))
    #         for i in circles:
    #             cv2.circle(self.image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #             cv2.circle(self.image, (i[0], i[1]), 2, (0, 0, 255), 3) # 中心点を描画
    #     # RGB画像を返す
    #     return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def template_matching(self, template_path, mask=None):
        # 画像を読み込む
        # template = cv2.imread(template_path, 0)  # テンプレートはグレースケールで読み込む
        # templateは2値化して読み込む
        template = cv2.imread(template_path, 0)
        template = cv2.imread('template/i8.png', cv2.IMREAD_UNCHANGED)
        _, alpha = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)
        # 透過部分を白にする
        template[:, :, 0] = cv2.bitwise_and(template[:, :, 0], alpha)
        template[:, :, 1] = cv2.bitwise_and(template[:, :, 1], alpha)
        template[:, :, 2] = cv2.bitwise_and(template[:, :, 2], alpha)
        # 透過部分を削除&グレースケールに変換
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
        # _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)    # 2値化
        self.template_shape = template.shape
        # テンプレートマッチングを実行
        if mask is not None:    # 前の検出部分の周辺のみを探索
            left, top = int(mask[0]), int(mask[1])
            near_pixel = 50
            near_mask = (top-near_pixel, top+self.template_shape[0]+near_pixel, left-near_pixel, left+self.template_shape[1]+near_pixel)
            # 画像サイズより大きくなる場合は，画像サイズに合わせる
            near_mask = (max(near_mask[0], 0), min(near_mask[1], self.gray.shape[1]), max(near_mask[2], 0), min(near_mask[3], self.gray.shape[1]))
            near_masked_gray = self.gray[near_mask[0]:near_mask[1], near_mask[2]:near_mask[3]]
            res = cv2.matchTemplate(near_masked_gray, template, cv2.TM_CCOEFF_NORMED)
        else:
            res = cv2.matchTemplate(self.masked_gray, template, cv2.TM_CCOEFF_NORMED)

        # 最も一致する位置を取得
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        # bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        print(max_val)
        if max_val < 0.5:
            # print("テンプレートが見つかりませんでした。")
            # return 1, 1, 0, 0
            return mask
        else:
            # center = top_left[0] + template.shape[1]//2, top_left[1] + template.shape[0]//2
            # return self.left+top_left[0], self.top+top_left[1], self.left+bottom_right[0], self.top+bottom_right[1] # 左上と右下の座標を返す
            if mask is not None:
                return top_left[0]+left-near_pixel, top_left[1]+top-near_pixel, self.template_shape[1], self.template_shape[0]
            else:
                return self.left+top_left[0], self.top+top_left[1], self.template_shape[1], self.template_shape[0] # 左上と幅，高さの座標を返す
            # return top_left[0], top_left[1], top_left[0]+template.shape[1], top_left[1]+template.shape[0] # template.shape[0]が高さ，template.shape[1]が幅
            # return top_left[0], top_left[1], template.shape[1], template.shape[0] # template.shape[0]が高さ，template.shape[1]が幅
        # # 元の画像に結果を反映
        # cv2.rectangle(frame, (self.left+top_left[0], self.top+top_left[1]), (self.left+bottom_right[0], self.top+bottom_right[1]), (0, 255, 0), 2)

    def draw_circles(self, center, r=35//2):
        # 検出された円を描画
        fixed_center = (center[0]+self.left, center[1]+self.top)

        cv2.circle(self.image, fixed_center, r, (0, 255, 0), 2)
        cv2.circle(self.image, fixed_center, 2, (0, 0, 255), 3) # 中心点を描画
        # RGB画像を返す
        # return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def draw_rectangle(self, bbox):
        # cv2.rectangle(self.image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        # p1 = (int(bbox[0]), int(bbox[1]))
        # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        int_bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        cv2.rectangle(self.image, int_bbox, (0, 255, 0), 2)


if __name__ == "__main__":
    image = cv2.imread("frame_500.jpg")
    frame = Frame(image)
    # circles = frame.detect_circles()
    # image_with_circles = frame.draw_circles(circles)

    horese8_bbox = frame.template_matching("./template/i8.png")
    frame.draw_rectangle(horese8_bbox)

    cv2.imshow('Frame', frame.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # plt.imshow(image_with_circles)
    # plt.show()
    # df = visualizer.detect_horses(circles)
    # print(df)