# yolo_c3.py YOLO V8 cluade version DEMO
'''
## pythonは3.11です。
streamlit==1.24.0
ultralytics==8.0.145
opencv-python==4.7.0.72
pillow==9.5.0
numpy==1.24.3
torch==2.0.1
torchvision==0.15.2
'''
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

class YOLOProcessVisualizer:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
    
    def preprocess_image(self, image):
        """画像の前処理"""
        img = np.array(image)
        img = cv2.resize(img, (640, 640))
        return img

    def extract_features(self, x):
        """特徴マップの抽出と可視化"""
        feature_maps = []
        channel_names = ['Red', 'Green', 'Blue']
        
        kernels = {
            'エッジ検出': np.array([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ]),
            '水平エッジ': np.array([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ]) / 4.0,
            '垂直エッジ': np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]) / 4.0
        }
        
        for channel_idx in range(3):
            channel = x[:, :, channel_idx]
            channel_name = channel_names[channel_idx]
            
            for kernel_name, kernel in kernels.items():
                feature = cv2.filter2D(channel.astype(np.float32), -1, kernel)
                
                feature = feature - feature.mean()
                if feature.std() != 0:
                    feature = feature / feature.std()
                
                feature = np.clip((feature + 1) * 127.5, 0, 255).astype(np.uint8)
                feature = cv2.equalizeHist(feature)
                
                feature_colored = cv2.applyColorMap(feature, cv2.COLORMAP_VIRIDIS)
                feature_colored = cv2.cvtColor(feature_colored, cv2.COLOR_BGR2RGB)
                
                feature_maps.append((f'{kernel_name} ({channel_name}チャンネル)', feature_colored))
        
        return feature_maps

    def visualize_grid_predictions(self, image, predictions, grid_size=7):
        """グリッドごとの予測を可視化"""
        img_h, img_w = image.shape[:2]
        cell_h, cell_w = img_h // grid_size, img_w // grid_size

        # グリッドの予測を表示
        grid_pred_img = image.copy()
        
        # グリッドを描画
        for i in range(grid_size + 1):
            y = i * cell_h
            cv2.line(grid_pred_img, (0, y), (img_w, y), (255, 255, 255), 1)
            x = i * cell_w
            cv2.line(grid_pred_img, (x, 0), (x, img_h), (255, 255, 255), 1)

        # 各グリッドセルからの予測を描画
        boxes = predictions.boxes
        for box in boxes:
            if box.conf.item() > 0.1:  # 低い閾値で表示
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                
                # ボックスの中心点を計算
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # 中心点が属するグリッドセルを特定
                grid_x = center_x // cell_w
                grid_y = center_y // cell_h
                
                # グリッドセルの中心を計算
                cell_center_x = (grid_x * cell_w) + (cell_w // 2)
                cell_center_y = (grid_y * cell_h) + (cell_h // 2)
                
                # グリッドセルから予測ボックスへの線を描画
                cv2.line(grid_pred_img, 
                        (cell_center_x, cell_center_y), 
                        (center_x, center_y),
                        (0, 255, 0), 2)
                
                # 予測ボックスを描画
                color = (int(255 * (1-conf)), 0, int(255 * conf))
                cv2.rectangle(grid_pred_img, (x1, y1), (x2, y2), color, 2)
                
                # 信頼度スコアを表示
                label = f'{conf:.2f}'
                cv2.putText(grid_pred_img, label,
                          (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, color, 2)
                
                # グリッドセルをハイライト
                grid_tl_x = grid_x * cell_w
                grid_tl_y = grid_y * cell_h
                cv2.rectangle(grid_pred_img,
                            (grid_tl_x, grid_tl_y),
                            (grid_tl_x + cell_w, grid_tl_y + cell_h),
                            (0, 255, 255), 2)

        return grid_pred_img

    def visualize_nms_process(self, image, predictions):
        """NMS前の検出ボックスを可視化"""
        pre_nms_img = image.copy()
        boxes = predictions.boxes
        
        for box in boxes:
            if box.conf.item() > 0.1:  # 低い閾値で表示
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                cls = box.cls.item()
                
                color = (0, int(255 * conf), int(255 * (1-conf)))
                cv2.rectangle(pre_nms_img, (x1, y1), (x2, y2), color, 2)
                
                label = f'{predictions.names[int(cls)]}: {conf:.2f}'
                cv2.putText(pre_nms_img, label,
                          (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, color, 2)
        
        return pre_nms_img

    def visualize_stages(self, image, grid_sizes=[7]):
        """YOLOの処理段階の可視化"""
        preprocessed = self.preprocess_image(image)
        stages = {
            'Original Image': np.array(image),
        }
        
        # 特徴マップの抽出
        feature_maps = self.extract_features(preprocessed)
        for name, fmap in feature_maps:
            stages[f'Feature Map: {name}'] = fmap
        
        # 予測の実行
        results = self.model.predict(preprocessed, verbose=False)[0]
        
        # グリッドベースの予測を可視化
        for grid_size in grid_sizes:
            grid_pred_img = self.visualize_grid_predictions(preprocessed, results, grid_size)
            stages[f'Grid Predictions ({grid_size}x{grid_size})'] = grid_pred_img
        
        # NMS前の予測を可視化
        stages['Pre-NMS Boxes'] = self.visualize_nms_process(preprocessed, results)
        
        # 最終検出結果
        result_img = preprocessed.copy()
        for box in results.boxes:
            if box.conf.item() > 0.5:  # 最終的な信頼度閾値
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = box.conf.item()
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{results.names[cls]}: {conf:.2f}'
                cv2.putText(result_img, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        stages['Final Detection'] = result_img
        return stages

def main():
    st.title("YOLO処理段階の可視化デモ")
    
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = YOLOProcessVisualizer()
    
    st.sidebar.title("設定")
    grid_size = st.sidebar.slider("グリッドサイズ", 5, 20, 7)
    show_feature_maps = st.sidebar.checkbox("特徴マップの表示", True)
    
    uploaded_file = st.file_uploader("画像をアップロードしてください", 
                                   type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize((640, 640))
        
        stages = st.session_state.visualizer.visualize_stages(
            image, grid_sizes=[grid_size])
        
        st.subheader("元画像")
        st.image(stages['Original Image'], caption="元の画像", use_column_width=True)
        
        if show_feature_maps:
            st.subheader("特徴マップ（RGB各チャンネルの特徴抽出）")
            feature_map_keys = [k for k in stages.keys() if 'Feature Map' in k]
    
            # チャンネルごとにグループ化
            for channel in ['Red', 'Green', 'Blue']:
                # 現在のチャンネルの特徴マップのキーを取得
                channel_keys = [k for k in feature_map_keys if f'({channel}' in k]
        
                st.write(f"**{channel}チャンネル**")
                cols = st.columns(3)
                for i, key in enumerate(channel_keys):
                    cols[i].image(
                        stages[key],
                        caption=key.replace('Feature Map: ', '').replace(f'({channel}チャンネル)', ''),
                        use_column_width=True
                    )

        
        
        st.subheader("グリッドごとの予測")
        st.image(stages[f'Grid Predictions ({grid_size}x{grid_size})'],
                caption=f"グリッドセルごとの予測結果 ({grid_size}x{grid_size}グリッド)\n" +
                       "黄色: 予測元のグリッドセル, 緑線: 予測の対応関係, " +
                       "赤→青: 信頼度（低→高）",
                use_column_width=True)
        
        st.subheader("NMS前の検出ボックス")
        st.image(stages['Pre-NMS Boxes'],
                caption="重複除去（NMS）適用前の全検出結果\n" +
                       "信頼度に応じて色が変化（低信頼度→高信頼度）",
                use_column_width=True)
        
        st.subheader("最終検出結果")
        st.image(stages['Final Detection'],
                caption="物体検出結果（NMS適用後、信頼度0.5以上）",
                use_column_width=True)

        st.markdown("""
        ### YOLOの処理段階の説明
        1. **特徴抽出**:
           - 画像からエッジや特徴的なパターンを抽出
           - RGB各チャンネルごとに異なる特徴を検出
        
        2. **グリッドごとの予測**:
           - 画像をグリッドセルに分割
           - 各グリッドセルが担当する領域の物体を予測
           - 予測には位置（バウンディングボックス）とその信頼度を含む
        
        3. **重複の除去（NMS）**:
           - 同じ物体に対する複数の予測を統合
           - 信頼度の高い検出を優先
           - 重複する検出を除去
        
        4. **最終検出**:
           - 信頼度の高い（0.5以上）検出のみを表示
           - クラス（物体の種類）と信頼度を表示
        """)

if __name__ == "__main__":
    main()
