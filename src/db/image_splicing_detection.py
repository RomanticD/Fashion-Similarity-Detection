import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


class ImageSplicingDetector:
    def __init__(self, vertical_threshold=280, horizontal_gap_threshold=1000, 
                 structure_threshold=0.92, gap_width=8):
        self.vertical_threshold = vertical_threshold  # 垂直边缘阈值提升至280
        self.horizontal_gap_threshold = horizontal_gap_threshold  # 水平间隔阈值提升至1000
        self.structure_threshold = structure_threshold  # 结构相似性阈值提升至0.92
        self.gap_width = gap_width

    def _check_structure_consistency(self, gray_img):
        """增强版结构检测"""
        height, width = gray_img.shape
        split_point = height // 2
        
        # 强化尺寸过滤条件（最小尺寸400x300）
        if split_point < 200 or width < 300 or height < 400:
            return False
            
        upper = gray_img[:split_point, :]
        lower = gray_img[split_point:split_point*2, :]
        
        # 新增多维度检测
        ssim = compare_ssim(upper, lower)
        color_corr = self._color_correlation(upper, lower)
        edge_match = self._edge_continuity(upper, lower)
        
        return (ssim < self.structure_threshold) and (color_corr < 0.95) and (edge_match < 0.85)

    def _color_correlation(self, upper, lower):
        """颜色分布相关性检测"""
        upper_hist = cv2.calcHist([upper],[0],None,[256],[0,256])
        lower_hist = cv2.calcHist([lower],[0],None,[256],[0,256])
        return cv2.compareHist(upper_hist, lower_hist, cv2.HISTCMP_CORREL)

    def _edge_continuity(self, upper, lower):
        """边缘连续性检测"""
        upper_edge = cv2.Canny(upper, 150, 250)
        lower_edge = cv2.Canny(lower, 150, 250)
        return cv2.matchShapes(upper_edge, lower_edge, cv2.CONTOURS_MATCH_I3, 0)

    def _check_horizontal_gap(self, gray_img):
        """优化后的水平间隔检测"""
        height, width = gray_img.shape
        # 增强检测条件：连续5行高亮像素且左右边缘一致
        for y in range(self.gap_width, height - self.gap_width):
            if all(np.mean(gray_img[y+i,:]) > 245 for i in range(-2,3)):  # 提高亮度阈值
                left_edge = np.mean(gray_img[y-2:y+3, :5]) > 240
                right_edge = np.mean(gray_img[y-2:y+3, -5:]) > 240
                if left_edge and right_edge:
                    return True
        return False

    def _analyze_frequency_domain(self, gray_img):
        """频域分析"""
        f = np.fft.fft2(gray_img)
        fshift = np.fft.fftshift(f)
        return 20 * np.log(np.abs(fshift))

    def _detect_vertical_edges(self, gray_img):
        """垂直边缘检测"""
        # 使用Sobel算子检测垂直边缘
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_y = np.absolute(sobel_y)
        scaled_sobel = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))
        
        # 应用阈值检测显著垂直边缘
        _, thresholded = cv2.threshold(scaled_sobel, self.vertical_threshold, 255, cv2.THRESH_BINARY)
        return thresholded

    def detect_image_splicing(self, image_path):
        """
        统一检测接口
        Args:
            image_path: 图片路径
        Returns:
            tuple: (is_spliced: bool, details: dict) 检测结果和详细信息
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图像")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 综合多个检测方法
        results = {
            "vertical_edges": self._detect_vertical_edges(gray),
            "horizontal_gap": self._check_horizontal_gap(gray),
            "frequency_analysis": self._analyze_frequency_domain(gray)
        }

        # 修改综合判断逻辑
        is_spliced = any([
            results["vertical_edges"].mean() > 250 and self._check_structure_consistency(gray),
            self._check_horizontal_gap(gray) and gray.shape[0] > 400,
            results["frequency_analysis"].mean() > 60
        ])
        
        return is_spliced, results
