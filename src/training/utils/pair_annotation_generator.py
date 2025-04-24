# Fashion-Similarity-Detection/training/utils/pair_annotation_generator.py
from pathlib import Path
import json

# ======================
# 可调整参数（集中在开头）
# ======================
BASE_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train")  # 项目根目录
PAIRS_ANNO_DIR = BASE_DIR / "pairs_annos"  # JSON存储目录
TRAINING_PAIRS_DIR = BASE_DIR / "training_pairs"  # 图片对目录
START_PAIR = 1  # 起始对号
END_PAIR = 1250  # 结束对号
DEFAULT_LABEL = 1.0  # 默认标签（针对整个照片对）

# 标签组配置（键为标签值，值为对应的对号列表）
LABEL_GROUPS = {
    0.9: [12, 21, 22, 23, 35, 45, 47, 58, 67, 68, 71, 73, 76, 81, 87, 109, 130, 132, 135, 136, 137, 141, 153, 155,
          158, 159, 162, 163, 165, 167, 168, 171, 195, 203, 204, 210, 222, 228, 236, 237, 239, 242, 247, 260, 277,
          278, 279, 280, 291, 297, 298, 299, 300, 310, 312, 313, 314, 315, 317, 320, 323, 329, 332, 340, 346, 350,
          356, 358, 368, 369, 377, 378, 383, 386, 388, 416, 423, 425, 433, 439, 454, 492, 494, 470, 471, 475, 491,
          496, 500, 503, 507, 518, 521, 527, 528, 529, 540, 546, 549, 550, 572, 574, 576, 578, 580, 582, 584, 588,
          591, 595, 597, 602, 603, 606, 611, 613, 616, 622, 625, 626, 628, 629, 631, 635, 639, 648, 652, 657, 660,
          669, 671, 673, 676, 678, 681, 686, 687, 692, 694, 695, 696, 697, 698, 699, 703, 705, 709, 710, 711, 715,
          718, 723, 738, 742, 743, 744, 747, 749, 751, 755, 757, 760, 761, 762, 764, 765, 769, 770, 771, 775, 777,
          779, 780, 782, 791, 794, 799, 801, 812, 820, 821, 822, 825, 827, 829, 831, 834, 835, 836, 839, 841, 847,
          849, 850, 851, 852, 861, 862, 865, 866, 867, 868, 869, 873, 878, 879, 880, 886, 894, 895, 900, 901, 903,
          914, 916, 917, 922, 925, 932, 934, 936, 939, 942, 944, 947, 952, 954, 957, 958, 959, 961, 962, 966, 969,
          971, 973, 976, 979, 984, 992, 993, 994, 995, 996, 998, 999, 1000, 1004, 1006, 1008, 1011, 1013, 1014, 1016,
          1020, 1025, 1031, 1039, 1050, 1052, 1053, 1055, 1059, 1062, 1067, 1069, 1072, 1073, 1080, 1085, 1088, 1095,
          1098, 1100, 1101, 1109, 1115, 1116, 1120, 1126, 1134, 1136, 1138, 1142, 1145, 1146, 1149, 1150, 1151, 1153,
          1157, 1160, 1162, 1163, 1164, 1167, 1173, 1174, 1178, 1179, 1181, 1182, 1186, 1187, 1188, 1189, 1192, 1199,
          1201, 1208, 1209, 1213, 1216, 1217, 1218, 1219, 1220, 1221, 1230, 1232, 1234, 1240, 1246, 1249, 1250],
    0.75: [19, 72, 190, 231, 261, 273, 316, 342, 348, 374, 375, 382, 403, 436, 448, 466, 493, 456, 484, 501, 520, 534,
           543, 564, 581, 594, 612, 632, 633, 656, 683, 690, 693, 702, 707, 713, 721, 725, 733, 737, 752, 754, 768,
           778, 806, 809, 816, 826, 830, 848, 856, 857, 859, 897, 898, 902, 915, 941, 987, 988, 1007, 1009, 1015,
           1022, 1084, 1094, 1107, 1111, 1128, 1147, 1191, 1236, 1244, 1245],
    0.5: [60, 61, 121, 209, 459, 510, 573, 620, 772, 1038, 1172, 1177, 1223, 1228, 1229],
    0: [24, 38, 39, 44, 881]
}

# 检查标签组内的对号是否在有效范围内
for label, pairs in LABEL_GROUPS.items():
    for pair_id in pairs:
        if not (START_PAIR <= pair_id <= END_PAIR):
            raise ValueError(f"Pair {pair_id} in label {label} is out of range [{START_PAIR}, {END_PAIR}]")


def generate_json():
    PAIRS_ANNO_DIR.mkdir(parents=True, exist_ok=True)
    for pair_id in range(START_PAIR, END_PAIR + 1):
        pair_dir = TRAINING_PAIRS_DIR / f"pair_{pair_id:04d}"
        if not pair_dir.exists():
            print(f"警告: pair_{pair_id:04d} 目录不存在，跳过该对")
            continue

        # 确定对级标签
        current_label = DEFAULT_LABEL
        for label, pairs in LABEL_GROUPS.items():
            if pair_id in pairs:
                current_label = label
                break  # 优先匹配第一个出现的标签（可根据需求调整匹配逻辑）

        # 构建图片信息（移除单个图片的label字段）
        images = []
        for img_idx, img_type in enumerate(["image_01", "image_02"], start=1):
            img_path = pair_dir / f"{img_type}.jpg"
            if not img_path.exists():
                print(f"警告: {img_type}.jpg 不存在，跳过该图片")
                continue
            images.append({
                "filename": f"{img_type}.jpg",
                "unique_id": f"pair_{pair_id:04d}_{img_idx}"
            })

        # 确保至少有一张图片存在（可根据需求调整为必须两张都存在）
        if not images:
            print(f"警告: pair_{pair_id:04d} 中无有效图片，跳过生成JSON")
            continue

        json_data = {
            "pair_id": f"pair_{pair_id:04d}",
            "label": current_label,
            "images": images
        }

        json_path = PAIRS_ANNO_DIR / f"pair_{pair_id:04d}.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"生成/更新 JSON: {json_path}")


if __name__ == "__main__":
    generate_json()
